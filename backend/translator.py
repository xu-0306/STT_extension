from __future__ import annotations

from typing import Dict, Optional, Tuple
import inspect
import threading

try:
    from .cache import LRUCache
except ImportError:  # Fallback when running as a script.
    from cache import LRUCache

DEFAULT_CACHE_SIZE = 512
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
SYSTEM_TRANSLATION_PROMPT = (
    "You are a translation engine. Follow the user's instructions exactly."
)

LANGUAGE_LABELS = {
    "auto": "auto-detected language",
    "en": "English (en)",
    "en-us": "English (en-US)",
    "en-gb": "English (en-GB)",
    "ja": "Japanese (ja)",
    "jp": "Japanese (ja)",
    "jpn": "Japanese (ja)",
    "zh": "Chinese (zh)",
    "zh-cn": "Simplified Chinese (zh-CN)",
    "zh-hans": "Simplified Chinese (zh-Hans)",
    "zh-tw": "Traditional Chinese (zh-TW)",
    "zh-hant": "Traditional Chinese (zh-Hant)",
    "zh-hk": "Traditional Chinese (zh-HK)",
    "ko": "Korean (ko)",
    "fr": "French (fr)",
    "de": "German (de)",
    "es": "Spanish (es)",
    "pt": "Portuguese (pt)",
    "it": "Italian (it)",
    "ru": "Russian (ru)",
    "id": "Indonesian (id)",
    "vi": "Vietnamese (vi)",
    "th": "Thai (th)",
}


def _describe_language(lang: Optional[str], fallback: str) -> str:
    if not lang:
        return fallback
    normalized = lang.replace("_", "-").strip().lower()
    return LANGUAGE_LABELS.get(normalized, normalized)


def _build_translation_prompt(
    text: str, source_lang: Optional[str], target_lang: str
) -> str:
    source_desc = _describe_language(source_lang, "source language")
    target_desc = _describe_language(target_lang, "target language")
    return (
        f"Translate the following {source_desc} text into {target_desc}. "
        f"Output only the translation in {target_desc}. "
        "Do not include the source text, explanations, or extra commentary. "
        "Do not mix other languages. "
        "Preserve meaning, punctuation, numbers, and proper nouns. "
        f"If the input is already in {target_desc}, return it unchanged.\n\n"
        f"{text}"
    )


class BaseTranslator:
    def __init__(self, target_language: str, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.target_language = target_language
        self._cache: LRUCache[Tuple[Optional[str], str, str], str] = LRUCache(
            cache_size
        )

    def translate(
        self, text: str, source_lang: Optional[str], target_lang: Optional[str] = None
    ) -> str:
        if not text:
            return ""
        target = target_lang or self.target_language
        key = (source_lang, target, text)
        cached, hit = self._cache.get(key)
        if hit:
            return cached or ""
        result = self._translate_impl(text, source_lang, target)
        self._cache.set(key, result)
        return result

    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        raise NotImplementedError


class NoopTranslator(BaseTranslator):
    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        return ""


class NLLBTranslator(BaseTranslator):
    def __init__(
        self,
        model: str,
        target_language: str,
        device: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        super().__init__(target_language, cache_size=cache_size)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self._model.to(self._device)
        self._model.eval()
        self._lock = threading.Lock()

    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        import torch

        src_lang = _map_nllb_lang(source_lang, target=False) or "eng_Latn"
        tgt_lang = _map_nllb_lang(target_lang, target=True) or "zho_Hant"
        self._tokenizer.src_lang = src_lang
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._lock, torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=256,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class OllamaTranslator(BaseTranslator):
    def __init__(
        self,
        model: str,
        host: str,
        target_language: str,
        keep_alive: Optional[object] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        super().__init__(target_language, cache_size=cache_size)
        import ollama

        self._client = ollama.Client(host=host)
        self._model = model
        self._keep_alive = _normalize_keep_alive(keep_alive)
        self._supports_keep_alive = _has_param(self._client.chat, "keep_alive")

    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        prompt = _build_translation_prompt(text, source_lang, target_lang)
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_TRANSLATION_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        if self._keep_alive is not None and self._supports_keep_alive:
            kwargs["keep_alive"] = self._keep_alive
        response = self._client.chat(**kwargs)
        return response["message"]["content"].strip()


class OpenAITranslator(BaseTranslator):
    def __init__(
        self,
        api_key: str,
        model: str,
        target_language: str,
        base_url: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        super().__init__(target_language, cache_size=cache_size)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        prompt = _build_translation_prompt(text, source_lang, target_lang)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_TRANSLATION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


def _map_nllb_lang(lang: Optional[str], target: bool = False) -> Optional[str]:
    if not lang:
        return None
    if "_" in lang:
        return lang
    normalized = lang.replace("_", "-").lower()
    if target:
        if normalized in {"zh-tw", "zh-hant", "zh-hk"}:
            return "zho_Hant"
        if normalized in {"zh-cn", "zh-hans", "zh"}:
            return "zho_Hans"
    mapping = {
        "en": "eng_Latn",
        "en-us": "eng_Latn",
        "en-gb": "eng_Latn",
        "ja": "jpn_Jpan",
        "jp": "jpn_Jpan",
        "jpn": "jpn_Jpan",
        "zh": "zho_Hans",
        "zh-cn": "zho_Hans",
        "zh-hans": "zho_Hans",
        "zh-tw": "zho_Hant",
        "zh-hant": "zho_Hant",
        "zh-hk": "zho_Hant",
    }
    return mapping.get(normalized)


def _normalize_cache_size(value: object, default: int) -> int:
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, size)


def _normalize_keep_alive(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, bool):
        return DEFAULT_OLLAMA_KEEP_ALIVE if value else 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return None


def _has_param(func, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def build_translator(cfg: Dict[str, object]) -> BaseTranslator:
    engine = str(cfg.get("engine", "nllb")).lower()
    target_language = str(cfg.get("target_language", "zh-TW"))
    cache_size = _normalize_cache_size(cfg.get("cache_size"), DEFAULT_CACHE_SIZE)

    if engine == "nllb":
        nllb_cfg = cfg.get("nllb", {}) if isinstance(cfg.get("nllb"), dict) else {}
        model = str(nllb_cfg.get("model", "facebook/nllb-200-distilled-600M"))
        device = nllb_cfg.get("device")
        return NLLBTranslator(
            model=model,
            device=device,
            target_language=target_language,
            cache_size=cache_size,
        )

    if engine == "ollama":
        ollama_cfg = cfg.get("ollama", {}) if isinstance(cfg.get("ollama"), dict) else {}
        model = str(ollama_cfg.get("model", "gemma3:4b"))
        host = str(ollama_cfg.get("host", "http://localhost:11434"))
        keep_alive = ollama_cfg.get("keep_alive")
        return OllamaTranslator(
            model=model,
            host=host,
            keep_alive=keep_alive,
            target_language=target_language,
            cache_size=cache_size,
        )

    if engine == "openai":
        openai_cfg = cfg.get("openai", {}) if isinstance(cfg.get("openai"), dict) else {}
        api_key = str(openai_cfg.get("api_key", ""))
        model = str(openai_cfg.get("model", "gpt-4o-mini"))
        base_url = str(openai_cfg.get("base_url", "")).strip() or None
        return OpenAITranslator(
            api_key=api_key,
            model=model,
            target_language=target_language,
            base_url=base_url,
            cache_size=cache_size,
        )

    return NoopTranslator(target_language=target_language, cache_size=cache_size)
