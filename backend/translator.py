from __future__ import annotations

from typing import Dict, Optional, Tuple
import threading

from cache import LRUCache

DEFAULT_CACHE_SIZE = 512


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
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        super().__init__(target_language, cache_size=cache_size)
        import ollama

        self._client = ollama.Client(host=host)
        self._model = model

    def _translate_impl(
        self, text: str, source_lang: Optional[str], target_lang: str
    ) -> str:
        source_label = source_lang or "source"
        prompt = (
            f"Translate the following {source_label} text into {target_lang}. "
            "Only output the translation.\n\n"
            f"{text}"
        )
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
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
        source_label = source_lang or "source"
        prompt = (
            f"Translate the following {source_label} text into {target_lang}. "
            "Only output the translation.\n\n"
            f"{text}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
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
        return OllamaTranslator(
            model=model,
            host=host,
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
