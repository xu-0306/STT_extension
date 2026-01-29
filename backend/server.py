import asyncio
import ipaddress
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import AudioProcessor, TranscriptionEngine

try:
    from .cache import LRUCache
    from .config import load_config
    from .translator import build_translator
except ImportError:  # Fallback when running as a script.
    from cache import LRUCache
    from config import load_config
    from translator import build_translator


def _get_server_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    server_cfg = cfg.get("server", {})
    return {
        "host": str(server_cfg.get("host", "127.0.0.1")),
        "port": int(server_cfg.get("port", 8765)),
    }


def _merge_config(base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not update:
        return dict(base)
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _normalize_stt_update(stt_update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not stt_update or not isinstance(stt_update, dict):
        return {}
    normalized = dict(stt_update)
    if "model" in normalized and "model_path" not in normalized:
        # Ensure explicit model selection does not keep a stale model_path.
        normalized["model_path"] = None
    if "model_cache_dir" in normalized and not normalized["model_cache_dir"]:
        normalized["model_cache_dir"] = None
    if "model_path" in normalized and not normalized["model_path"]:
        normalized["model_path"] = None
    return normalized


def _resolve_model_target(stt_cfg: Dict[str, Any]) -> Optional[Path]:
    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return None
    model_path = stt_cfg.get("model_path")
    if model_path:
        return Path(model_path)
    model_cache_dir = stt_cfg.get("model_cache_dir")
    if not model_cache_dir:
        return None
    return Path(model_cache_dir) / f"{model_name}.pt"


def _download_model_file(
    url: str,
    target: Path,
    progress_cb=None,
) -> None:
    import urllib.request

    tmp_path = target.with_suffix(target.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            total_size = None
            try:
                total_header = response.getheader("Content-Length")
                if total_header:
                    total_size = int(total_header)
            except (TypeError, ValueError):
                total_size = None
            downloaded = 0
            if progress_cb:
                progress_cb(downloaded, total_size)
            while True:
                chunk = response.read(MODEL_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total_size)
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def _ensure_model_available(
    stt_cfg: Dict[str, Any],
    websocket: WebSocket,
    download_lock: asyncio.Lock,
) -> Dict[str, Any]:
    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return stt_cfg
    url = WHISPER_MODEL_URLS.get(model_name)
    if not url:
        return stt_cfg
    target = _resolve_model_target(stt_cfg)
    if target is None:
        return stt_cfg
    if target.exists():
        updated = dict(stt_cfg)
        updated["model_path"] = str(target)
        return updated
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        await _safe_send(
            websocket,
            {
                "type": "error",
                "message": f"Model cache dir error: {type(exc).__name__}",
            },
        )
        raise
    async with download_lock:
        if target.exists():
            updated = dict(stt_cfg)
            updated["model_path"] = str(target)
            return updated
        await _safe_send(
            websocket,
            {"type": "status", "message": f"Downloading Whisper model: {model_name}"},
        )
        loop = asyncio.get_running_loop()
        last_progress = {"percent": -1, "bytes": 0, "ts": 0.0}

        def _format_mb(value: int) -> str:
            return f"{value / (1024 * 1024):.1f}"

        def _report_progress(downloaded: int, total: Optional[int]) -> None:
            now = time.monotonic()
            if total and total > 0:
                percent = int(downloaded * 100 / total)
                if percent <= last_progress["percent"] and now - last_progress["ts"] < 1.0:
                    return
                last_progress["percent"] = percent
                message = (
                    f"Downloading Whisper model: {model_name} "
                    f"({percent}%, {_format_mb(downloaded)}/{_format_mb(total)} MB)"
                )
            else:
                if (
                    downloaded - last_progress["bytes"] < 5 * 1024 * 1024
                    and now - last_progress["ts"] < 1.0
                ):
                    return
                last_progress["bytes"] = downloaded
                message = (
                    f"Downloading Whisper model: {model_name} "
                    f"({_format_mb(downloaded)} MB)"
                )
            last_progress["ts"] = now

            def _emit() -> None:
                asyncio.create_task(
                    _safe_send(websocket, {"type": "status", "message": message})
                )

            loop.call_soon_threadsafe(_emit)

        try:
            await asyncio.to_thread(_download_model_file, url, target, _report_progress)
        except Exception as exc:
            await _safe_send(
                websocket,
                {
                    "type": "error",
                    "message": f"Model download failed: {type(exc).__name__}",
                },
            )
            raise
        await _safe_send(
            websocket,
            {"type": "status", "message": f"Model downloaded: {model_name}"},
        )
    updated = dict(stt_cfg)
    updated["model_path"] = str(target)
    return updated


def _build_engine_kwargs(stt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    simul_cfg = stt_cfg.get("simulstreaming", {}) if isinstance(stt_cfg, dict) else {}
    max_context_tokens = simul_cfg.get("max_context_tokens")
    if max_context_tokens is None:
        max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    return {
        "model_size": stt_cfg.get("model", "medium"),
        "lan": stt_cfg.get("language", "auto"),
        "backend_policy": stt_cfg.get("backend_policy", "simulstreaming"),
        "backend": stt_cfg.get("backend", "auto"),
        "min_chunk_size": float(stt_cfg.get("min_chunk_size", 0.1)),
        "buffer_trimming": stt_cfg.get("buffer_trimming", "segment"),
        "buffer_trimming_sec": float(stt_cfg.get("buffer_trimming_sec", 15.0)),
        "confidence_validation": bool(stt_cfg.get("confidence_validation", False)),
        "pcm_input": bool(stt_cfg.get("pcm_input", False)),
        "vad": bool(stt_cfg.get("vad", True)),
        "vac": bool(stt_cfg.get("vac", True)),
        "vac_chunk_size": float(stt_cfg.get("vac_chunk_size", 0.04)),
        "model_cache_dir": stt_cfg.get("model_cache_dir"),
        "model_dir": stt_cfg.get("model_dir"),
        "model_path": stt_cfg.get("model_path"),
        "lora_path": stt_cfg.get("lora_path"),
        "disable_fast_encoder": bool(simul_cfg.get("disable_fast_encoder", False)),
        "custom_alignment_heads": simul_cfg.get("custom_alignment_heads"),
        "frame_threshold": int(simul_cfg.get("frame_threshold", 25)),
        "beams": int(simul_cfg.get("beams", 1)),
        "decoder_type": simul_cfg.get("decoder_type"),
        "audio_max_len": float(simul_cfg.get("audio_max_len", 30.0)),
        "audio_min_len": float(simul_cfg.get("audio_min_len", 0.0)),
        "cif_ckpt_path": simul_cfg.get("cif_ckpt_path"),
        "never_fire": bool(simul_cfg.get("never_fire", False)),
        "init_prompt": simul_cfg.get("init_prompt"),
        "static_init_prompt": simul_cfg.get("static_init_prompt"),
        "max_context_tokens": max_context_tokens,
        "target_language": "",
    }


async def _close_results_generator(results_generator) -> None:
    closer = getattr(results_generator, "aclose", None)
    if not callable(closer):
        return
    try:
        await closer()
    except Exception:
        pass


def _release_torch_cache() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _pick_latest_segment(lines: list[Any]) -> Optional[Any]:
    for line in reversed(lines):
        if getattr(line, "text", None):
            return line
    return None


def _guess_language(text: str, fallback: Optional[str]) -> Optional[str]:
    if fallback and fallback != "auto":
        return fallback
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30ff or 0x31f0 <= code <= 0x31ff:
            return "ja"
        if 0x4e00 <= code <= 0x9fff:
            return "zh"
    if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in text):
        return "en"
    return fallback


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\ufffd", "").replace("\u0000", "")
    return cleaned.strip()


MAX_DISPLAY_CHARS = 260
MAX_SENTENCES_DEFAULT = 2
MAX_SENTENCES_CJK = 1
DEFAULT_TRANSLATION_DEBOUNCE_MS = 300
DEFAULT_MAX_CONTEXT_TOKENS = 64
DEFAULT_STALL_TIMEOUT_SEC = 15.0
DEFAULT_STALL_CHECK_INTERVAL_SEC = 5.0
DEFAULT_SEGMENT_MAX_CHARS = 160
DEFAULT_SEGMENT_MAX_MS = 4000
INITIAL_CONFIG_TIMEOUT_SEC = 0.5
SENTENCE_ENDINGS = {".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026"}
CLOSING_PUNCTUATION = {")", "]", "}", "\"", "'"}
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]")

MODEL_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
WHISPER_MODEL_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _split_sentences(text: str) -> list[str]:
    segments: list[str] = []
    start = 0
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch in SENTENCE_ENDINGS:
            end = idx + 1
            while end < len(text) and text[end] in CLOSING_PUNCTUATION:
                end += 1
            part = text[start:end].strip()
            if part:
                segments.append(part)
            start = end
            idx = end
            continue
        idx += 1
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _dedupe_repeated_sentences(text: str, max_repeats: int) -> str:
    if not text or max_repeats < 1:
        return text
    segments = _split_sentences(text)
    if not segments:
        return text
    deduped: list[str] = []
    last_key: Optional[str] = None
    repeat = 0
    for segment in segments:
        normalized = _normalize_text(segment).strip(")]}\"'")
        if not normalized:
            continue
        key = normalized.casefold()
        if key == last_key:
            repeat += 1
        else:
            repeat = 1
            last_key = key
        if repeat > max_repeats:
            continue
        deduped.append(segment.strip())
    if not deduped:
        return ""
    return " ".join(deduped).strip()


def _trim_text(text: str, max_chars: int) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    trimmed = text[-max_chars:]
    return trimmed.lstrip(" \t\n\r.,;:!?-")


def _compact_text(
    text: str,
    max_chars: int,
    max_sentences_default: int,
    max_sentences_cjk: int,
) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    if not max_chars or len(normalized) <= max_chars:
        return normalized
    sentence_limit = (
        max_sentences_cjk if CJK_RE.search(normalized) else max_sentences_default
    )
    segments = _split_sentences(normalized)
    if len(segments) > sentence_limit:
        tail = " ".join(segments[-sentence_limit:])
        return _trim_text(tail, max_chars)
    return _trim_text(normalized, max_chars)


def _merge_utterance_text(current: str, incoming: str) -> str:
    if not current:
        return incoming
    if incoming.startswith(current):
        return incoming
    if current.startswith(incoming):
        return current
    return f"{current} {incoming}".strip()


def _translation_cache_key(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _normalize_cache_size(value: object, default: int) -> int:
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, size)


def _normalize_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_non_negative_int(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _normalize_timeout(value: object, default: float) -> float:
    try:
        timeout = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if timeout < 0:
        return default
    return timeout


def _normalize_delay_ms(value: object, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _is_local_client(websocket: WebSocket) -> bool:
    client = websocket.client
    if not client or not client.host:
        return False
    host = client.host
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


async def _resolve_translator(cache: LRUCache[str, Any], cfg: Dict[str, Any]):
    key = _translation_cache_key(cfg)
    cached, hit = cache.get(key)
    if hit:
        return cached
    translator = await asyncio.to_thread(build_translator, cfg)
    cache.set(key, translator)
    return translator


@dataclass
class TranslationRequest:
    text: str
    language: Optional[str]
    is_final: bool
    seq: int
    version: int


class TranslationSession:
    def __init__(
        self,
        cfg: Dict[str, Any],
        default_lang: Optional[str],
        translator,
        subtitle_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_lang = default_lang
        self.cfg = dict(cfg)
        self.translator = translator
        self.translate_partials = bool(self.cfg.get("partial", False))
        subtitle_cfg = subtitle_cfg or {}
        self.max_chars = _normalize_positive_int(
            subtitle_cfg.get("max_chars"), MAX_DISPLAY_CHARS
        )
        self.max_sentences_default = _normalize_positive_int(
            subtitle_cfg.get("max_sentences_default"), MAX_SENTENCES_DEFAULT
        )
        self.max_sentences_cjk = _normalize_positive_int(
            subtitle_cfg.get("max_sentences_cjk"), MAX_SENTENCES_CJK
        )
        self.max_repeat_sentences = _normalize_positive_int(
            subtitle_cfg.get("max_repeat_sentences"), 1
        )
        self.seq = 0
        self.pending_seq: Optional[int] = None
        self.translation_task: Optional[asyncio.Task[None]] = None
        self.debounce_ms = _normalize_delay_ms(
            self.cfg.get("debounce_ms"), DEFAULT_TRANSLATION_DEBOUNCE_MS
        )
        self.debounce_version = 0
        self.debounce_task: Optional[asyncio.Task[None]] = None
        self.utterance_text = ""
        self.utterance_language: Optional[str] = None
        self.pending_request: Optional[TranslationRequest] = None
        self.request_version = 0
        self.segment_max_chars = _normalize_non_negative_int(
            self.cfg.get("segment_max_chars"), DEFAULT_SEGMENT_MAX_CHARS
        )
        self.segment_max_ms = _normalize_delay_ms(
            self.cfg.get("segment_max_ms"), DEFAULT_SEGMENT_MAX_MS
        )
        self.segment_start_ts: Optional[float] = None


class SttSession:
    def __init__(
        self,
        cfg: Dict[str, Any],
        engine: TranscriptionEngine,
        audio_processor: AudioProcessor,
    ) -> None:
        self.cfg = dict(cfg)
        self.default_language = str(self.cfg.get("language", "auto"))
        self.engine = engine
        self.audio_processor = audio_processor
        self.first_audio_ts: Optional[float] = None
        self.last_audio_ts = time.monotonic()
        self.last_result_ts = time.monotonic()
        self.result_seen = False
        self.reset_lock = asyncio.Lock()


async def _safe_send(websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    try:
        await websocket.send_text(json.dumps(payload))
        return True
    except (WebSocketDisconnect, RuntimeError, OSError):
        return False


async def _send_subtitle(
    websocket: WebSocket,
    text: str,
    translated: str,
    language: Optional[str],
    is_final: bool,
    seq: int,
) -> bool:
    return await _safe_send(
        websocket,
        {
            "type": "subtitle",
            "original": text,
            "translated": translated,
            "language": language,
            "timestamp": int(time.time() * 1000),
            "final": is_final,
            "seq": seq,
        },
    )


def _ensure_pending_seq(session: TranslationSession) -> int:
    if session.pending_seq is None:
        session.pending_seq = session.seq + 1
    return session.pending_seq


def _current_seq(session: TranslationSession) -> int:
    return session.pending_seq if session.pending_seq is not None else session.seq


def _should_refresh_engine(current: Dict[str, Any], updated: Dict[str, Any]) -> bool:
    return current != updated


async def _translate_text(
    session: TranslationSession,
    text: str,
    language: Optional[str],
    websocket: WebSocket,
) -> str:
    translation_timeout = _normalize_timeout(session.cfg.get("timeout_sec"), 8.0)
    if translation_timeout > 0:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(session.translator.translate, text, language),
                timeout=translation_timeout,
            )
        except asyncio.TimeoutError:
            return ""
        except Exception as exc:
            await _safe_send(
                websocket,
                {"type": "error", "message": f"Translation error: {type(exc).__name__}"},
            )
            return ""
    try:
        return await asyncio.to_thread(session.translator.translate, text, language)
    except Exception as exc:
        await _safe_send(
            websocket,
            {"type": "error", "message": f"Translation error: {type(exc).__name__}"},
        )
        return ""


async def _translate_and_send(
    websocket: WebSocket,
    session: TranslationSession,
    request: TranslationRequest,
) -> None:
    translated = await _translate_text(
        session, request.text, request.language, websocket
    )
    if request.version != session.request_version:
        return
    if not translated:
        if request.seq == _current_seq(session):
            session.seq = request.seq
            session.pending_seq = None
            session.utterance_text = ""
            session.utterance_language = None
            session.segment_start_ts = None
        return
    if request.seq != _current_seq(session):
        return
    await _send_subtitle(
        websocket,
        request.text,
        translated,
        request.language,
        request.is_final,
        request.seq,
    )
    if request.seq == _current_seq(session):
        session.seq = request.seq
        session.pending_seq = None
        session.utterance_text = ""
        session.utterance_language = None
        session.segment_start_ts = None


async def _translation_worker(
    websocket: WebSocket,
    session: TranslationSession,
) -> None:
    while True:
        request = session.pending_request
        if request is None:
            return
        session.pending_request = None
        try:
            await _translate_and_send(websocket, session, request)
        except asyncio.CancelledError:
            return


def _queue_translation(
    websocket: WebSocket,
    session: TranslationSession,
    text: str,
    language: Optional[str],
    is_final: bool,
    seq: int,
) -> None:
    session.request_version += 1
    session.pending_request = TranslationRequest(
        text=text,
        language=language,
        is_final=is_final,
        seq=seq,
        version=session.request_version,
    )
    if not session.translation_task or session.translation_task.done():
        session.translation_task = asyncio.create_task(
            _translation_worker(websocket, session)
        )


def _cancel_translation_debounce(session: TranslationSession) -> None:
    session.debounce_version += 1
    if session.debounce_task and not session.debounce_task.done():
        session.debounce_task.cancel()
    session.debounce_task = None


async def _debounced_translation_start(
    websocket: WebSocket,
    session: TranslationSession,
    version: int,
) -> None:
    delay_ms = session.debounce_ms
    if delay_ms > 0:
        try:
            await asyncio.sleep(delay_ms / 1000.0)
        except asyncio.CancelledError:
            return
    if version != session.debounce_version:
        return
    text = session.utterance_text
    if not text:
        return
    seq = _ensure_pending_seq(session)
    _queue_translation(
        websocket,
        session,
        text,
        session.utterance_language,
        True,
        seq,
    )


async def _schedule_translation_debounce(
    websocket: WebSocket,
    session: TranslationSession,
) -> None:
    session.debounce_version += 1
    version = session.debounce_version
    if session.debounce_task and not session.debounce_task.done():
        session.debounce_task.cancel()
    session.debounce_task = asyncio.create_task(
        _debounced_translation_start(websocket, session, version)
    )


def _should_force_segment(session: TranslationSession, now: float) -> bool:
    if (
        session.segment_max_chars > 0
        and len(session.utterance_text) >= session.segment_max_chars
    ):
        return True
    if session.segment_start_ts is None or session.segment_max_ms <= 0:
        return False
    elapsed_ms = (now - session.segment_start_ts) * 1000.0
    return elapsed_ms >= session.segment_max_ms


async def _apply_stt_update(
    stt_session: SttSession,
    translation_session: TranslationSession,
    stt_update: Optional[Dict[str, Any]],
    websocket: WebSocket,
) -> None:
    normalized_update = _normalize_stt_update(stt_update)
    if not normalized_update:
        return
    current_cfg = stt_session.cfg
    merged_cfg = _merge_config(current_cfg, normalized_update)
    if not _should_refresh_engine(current_cfg, merged_cfg):
        return
    try:
        merged_cfg = await _ensure_model_available(
            merged_cfg, websocket, app.state.model_download_lock
        )
        engine_kwargs = _build_engine_kwargs(merged_cfg)
        engine_kwargs["pcm_input"] = bool(merged_cfg.get("pcm_input", False))
        new_engine = TranscriptionEngine(**engine_kwargs)
    except Exception as exc:
        await _safe_send(
            websocket,
            {
                "type": "error",
                "message": f"STT config error: {type(exc).__name__}",
            },
        )
        return
    stt_session.cfg = merged_cfg
    stt_session.default_language = str(merged_cfg.get("language", "auto"))
    stt_session.engine = new_engine
    stt_session.audio_processor.transcription_engine = new_engine
    stt_session.last_result_ts = time.monotonic()
    stt_session.result_seen = False
    translation_session.default_lang = stt_session.default_language
    await _safe_send(
        websocket,
        {
            "type": "status",
            "message": "stt updated",
        },
    )


async def _handle_results(
    websocket: WebSocket,
    results_generator,
    session: TranslationSession,
    stt_session: SttSession,
) -> None:
    last_final = ""
    last_partial = ""
    async for response in results_generator:
        if getattr(response, "error", ""):
            await _safe_send(
                websocket,
                {
                    "type": "error",
                    "message": response.error,
                },
            )
            continue

        text = ""
        language = None
        is_final = False
        segment = _pick_latest_segment(response.lines or [])
        if segment is not None:
            text = segment.text.strip()
            language = getattr(segment, "detected_language", None)
            is_final = True
        elif response.buffer_transcription:
            text = response.buffer_transcription.strip()
            is_final = False

        text = _sanitize_text(text)
        if not text:
            continue
        text = _dedupe_repeated_sentences(text, session.max_repeat_sentences)
        if not text:
            continue

        display_text = _compact_text(
            text,
            session.max_chars,
            session.max_sentences_default,
            session.max_sentences_cjk,
        )
        if not display_text:
            continue
        stt_session.last_result_ts = time.monotonic()
        stt_session.result_seen = True

        language = _guess_language(display_text, language or session.default_lang)
        if is_final:
            if display_text == last_final:
                continue
            last_final = display_text
            if not session.utterance_text:
                session.segment_start_ts = time.monotonic()
            merged_text = _merge_utterance_text(session.utterance_text, display_text)
            session.utterance_text = _compact_text(
                merged_text,
                session.max_chars,
                session.max_sentences_default,
                session.max_sentences_cjk,
            )
            language = _guess_language(
                session.utterance_text,
                language or session.default_lang,
            )
            session.utterance_language = language
            seq = _ensure_pending_seq(session)
            await _send_subtitle(
                websocket,
                session.utterance_text,
                "",
                language,
                True,
                seq,
            )
            if _should_force_segment(session, time.monotonic()):
                flush_text = session.utterance_text
                flush_lang = session.utterance_language
                _cancel_translation_debounce(session)
                _queue_translation(
                    websocket,
                    session,
                    flush_text,
                    flush_lang,
                    True,
                    seq,
                )
                session.utterance_text = ""
                session.utterance_language = None
                session.segment_start_ts = None
            else:
                await _schedule_translation_debounce(websocket, session)
        else:
            if display_text == last_partial:
                continue
            last_partial = display_text
            seq = _ensure_pending_seq(session)
            await _send_subtitle(
                websocket,
                display_text,
                "",
                language,
                False,
                seq,
            )
            if session.translate_partials:
                if not session.utterance_text:
                    session.segment_start_ts = time.monotonic()
                session.utterance_text = display_text
                session.utterance_language = language
                if _should_force_segment(session, time.monotonic()):
                    flush_text = session.utterance_text
                    flush_lang = session.utterance_language
                    _cancel_translation_debounce(session)
                    _queue_translation(
                        websocket,
                        session,
                        flush_text,
                        flush_lang,
                        False,
                        seq,
                    )
                    session.utterance_text = ""
                    session.utterance_language = None
                    session.segment_start_ts = None
                else:
                    await _schedule_translation_debounce(websocket, session)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.server_config = _get_server_config(cfg)
    stt_cfg = cfg.get("stt", {})
    translation_cfg = cfg.get("translation", {})
    subtitle_cfg = cfg.get("subtitle", {})
    translator_cache_size = _normalize_cache_size(
        translation_cfg.get("translator_cache_size", 4),
        4,
    )

    app.state.default_stt_cfg = stt_cfg
    app.state.translation_cfg = translation_cfg
    app.state.subtitle_cfg = {
        "max_chars": _normalize_positive_int(
            subtitle_cfg.get("max_chars"), MAX_DISPLAY_CHARS
        ),
        "max_sentences_default": _normalize_positive_int(
            subtitle_cfg.get("max_sentences_default"), MAX_SENTENCES_DEFAULT
        ),
        "max_sentences_cjk": _normalize_positive_int(
            subtitle_cfg.get("max_sentences_cjk"), MAX_SENTENCES_CJK
        ),
        "max_repeat_sentences": _normalize_positive_int(
            subtitle_cfg.get("max_repeat_sentences"), 1
        ),
    }
    app.state.translator_cache = LRUCache(translator_cache_size)
    app.state.model_download_lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if not _is_local_client(websocket):
        await websocket.accept()
        await _safe_send(
            websocket,
            {"type": "error", "message": "Local connections only"},
        )
        await websocket.close(code=1008)
        return
    await websocket.accept()
    stt_cfg = dict(app.state.default_stt_cfg or {})
    translation_cfg = dict(app.state.translation_cfg or {})
    pending_message: Optional[Dict[str, Any]] = None
    try:
        initial_message = await asyncio.wait_for(
            websocket.receive(), timeout=INITIAL_CONFIG_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        initial_message = None
    if initial_message:
        if initial_message.get("type") == "websocket.disconnect":
            return
        if "text" in initial_message and initial_message["text"] is not None:
            try:
                payload = json.loads(initial_message["text"])
            except json.JSONDecodeError:
                pending_message = initial_message
            else:
                if payload.get("type") == "config":
                    translation_update = payload.get("translation")
                    if translation_update:
                        translation_cfg = _merge_config(
                            translation_cfg, translation_update
                        )
                    stt_update = _normalize_stt_update(payload.get("stt"))
                    if stt_update:
                        stt_cfg = _merge_config(stt_cfg, stt_update)
                else:
                    pending_message = initial_message
        else:
            pending_message = initial_message

    try:
        stt_cfg = await _ensure_model_available(
            stt_cfg, websocket, app.state.model_download_lock
        )
    except Exception:
        await websocket.close(code=1011)
        return
    engine_kwargs = _build_engine_kwargs(stt_cfg)
    engine_kwargs["pcm_input"] = bool(stt_cfg.get("pcm_input", False))
    stt_engine = TranscriptionEngine(**engine_kwargs)
    translator = await _resolve_translator(app.state.translator_cache, translation_cfg)
    audio_processor = AudioProcessor(
        transcription_engine=stt_engine,
    )
    stt_session = SttSession(stt_cfg, stt_engine, audio_processor)
    session = TranslationSession(
        translation_cfg,
        stt_session.default_language,
        translator,
        app.state.subtitle_cfg,
    )
    results_generator = await audio_processor.create_tasks()
    results_task = asyncio.create_task(
        _handle_results(websocket, results_generator, session, stt_session)
    )
    stall_timeout = _normalize_timeout(
        stt_cfg.get("stall_timeout_sec"), DEFAULT_STALL_TIMEOUT_SEC
    )
    stall_check_interval = _normalize_timeout(
        stt_cfg.get("stall_check_interval_sec"),
        DEFAULT_STALL_CHECK_INTERVAL_SEC,
    )

    async def _reset_stt_engine(reason: str, rebuild_engine: bool = True) -> None:
        nonlocal audio_processor, stt_engine, results_generator, results_task
        async with stt_session.reset_lock:
            try:
                if session.translation_task and not session.translation_task.done():
                    session.translation_task.cancel()
                    try:
                        await session.translation_task
                    except asyncio.CancelledError:
                        pass
                if session.debounce_task and not session.debounce_task.done():
                    session.debounce_task.cancel()
                    try:
                        await session.debounce_task
                    except asyncio.CancelledError:
                        pass
                session.utterance_text = ""
                session.utterance_language = None
                session.pending_seq = None
                session.pending_request = None
                session.request_version = 0
                session.segment_start_ts = None
                if not results_task.done():
                    results_task.cancel()
                    try:
                        await results_task
                    except asyncio.CancelledError:
                        pass
                await _close_results_generator(results_generator)
                await audio_processor.cleanup()
                if rebuild_engine:
                    engine_kwargs = _build_engine_kwargs(stt_session.cfg)
                    engine_kwargs["pcm_input"] = bool(
                        stt_session.cfg.get("pcm_input", False)
                    )
                    stt_engine = TranscriptionEngine(**engine_kwargs)
                audio_processor = AudioProcessor(transcription_engine=stt_engine)
                stt_session.engine = stt_engine
                stt_session.audio_processor = audio_processor
                stt_session.first_audio_ts = None
                stt_session.last_audio_ts = time.monotonic()
                stt_session.last_result_ts = time.monotonic()
                stt_session.result_seen = False
                _release_torch_cache()
                results_generator = await audio_processor.create_tasks()
                results_task = asyncio.create_task(
                    _handle_results(websocket, results_generator, session, stt_session)
                )
                await _safe_send(
                    websocket,
                    {"type": "status", "message": f"stt reset: {reason}"},
                )
            except Exception as exc:
                await _safe_send(
                    websocket,
                    {"type": "error", "message": f"STT reset failed: {type(exc).__name__}"},
                )

    async def _stt_watchdog() -> None:
        if stall_check_interval <= 0 or stall_timeout <= 0:
            return
        while True:
            try:
                await asyncio.sleep(stall_check_interval)
            except asyncio.CancelledError:
                return
            if stt_session.first_audio_ts is None:
                continue
            if stt_session.reset_lock.locked():
                continue
            now = time.monotonic()
            if now - stt_session.last_audio_ts > stall_timeout:
                continue
            if stt_session.result_seen:
                if now - stt_session.last_result_ts > stall_timeout:
                    await _reset_stt_engine("stall", rebuild_engine=False)
            else:
                if now - stt_session.first_audio_ts > stall_timeout * 2:
                    await _reset_stt_engine("no results", rebuild_engine=False)

    watchdog_task = asyncio.create_task(_stt_watchdog())

    try:
        await _safe_send(websocket, {"type": "status", "message": "connected"})
        while True:
            try:
                if pending_message is not None:
                    message = pending_message
                    pending_message = None
                else:
                    message = await websocket.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                now = time.monotonic()
                stt_session.last_audio_ts = now
                if stt_session.first_audio_ts is None:
                    stt_session.first_audio_ts = now
                try:
                    async with stt_session.reset_lock:
                        await audio_processor.process_audio(message["bytes"])
                except Exception as exc:
                    await _safe_send(
                        websocket,
                        {
                            "type": "error",
                            "message": f"STT audio error: {type(exc).__name__}",
                        },
                    )
                    await _reset_stt_engine("audio error")
                continue
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = payload.get("type")
                if msg_type == "config":
                    translation_update = payload.get("translation")
                    if translation_update:
                        merged = _merge_config(session.cfg, translation_update)
                        session.translator = await _resolve_translator(
                            app.state.translator_cache, merged
                        )
                        session.cfg = merged
                        session.translate_partials = bool(
                            session.cfg.get("partial", False)
                        )
                        session.debounce_ms = _normalize_delay_ms(
                            session.cfg.get("debounce_ms"),
                            DEFAULT_TRANSLATION_DEBOUNCE_MS,
                        )
                        session.segment_max_chars = _normalize_non_negative_int(
                            session.cfg.get("segment_max_chars"),
                            DEFAULT_SEGMENT_MAX_CHARS,
                        )
                        session.segment_max_ms = _normalize_delay_ms(
                            session.cfg.get("segment_max_ms"),
                            DEFAULT_SEGMENT_MAX_MS,
                        )
                        session.segment_start_ts = None
                        await _safe_send(
                            websocket,
                            {
                                "type": "status",
                                "message": "translation updated",
                            },
                        )
                    stt_update = payload.get("stt")
                    if stt_update:
                        await _apply_stt_update(
                            stt_session,
                            session,
                            stt_update,
                            websocket,
                        )
                elif msg_type == "test":
                    if payload.get("target") == "translation":
                        test_cfg = _merge_config(
                            session.cfg, payload.get("translation")
                        )
                        try:
                            test_translator = await _resolve_translator(
                                app.state.translator_cache, test_cfg
                            )
                            result = await asyncio.to_thread(
                                test_translator.translate, "Hello world", "en"
                            )
                            await _safe_send(
                                websocket,
                                {
                                    "type": "test_result",
                                    "target": "translation",
                                    "ok": True,
                                    "message": f"Translation ok: {result[:60]}",
                                },
                            )
                        except Exception as exc:
                            await _safe_send(
                                websocket,
                                {
                                    "type": "test_result",
                                    "target": "translation",
                                    "ok": False,
                                    "message": str(exc),
                                },
                            )
                elif msg_type == "ping":
                    continue
    except WebSocketDisconnect:
        pass
    finally:
        if not watchdog_task.done():
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        if not results_task.done():
            results_task.cancel()
            try:
                await results_task
            except asyncio.CancelledError:
                pass
        if session.translation_task and not session.translation_task.done():
            session.translation_task.cancel()
            try:
                await session.translation_task
            except asyncio.CancelledError:
                pass
        if session.debounce_task and not session.debounce_task.done():
            session.debounce_task.cancel()
            try:
                await session.debounce_task
            except asyncio.CancelledError:
                pass
        await audio_processor.cleanup()
        # No global WS lock; each connection is handled independently.


def main() -> None:
    import argparse
    import os
    import uvicorn

    parser = argparse.ArgumentParser(description="WhisperLiveKit backend server")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    args = parser.parse_args()

    if args.config:
        os.environ["STT_CONFIG_PATH"] = args.config

    cfg = load_config(args.config)
    server_cfg = _get_server_config(cfg)
    uvicorn.run(
        app,
        host=args.host or server_cfg["host"],
        port=args.port or server_cfg["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
