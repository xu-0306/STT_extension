import asyncio
import ipaddress
import json
import re
import time
from contextlib import asynccontextmanager
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
DEFAULT_MAX_CONTEXT_TOKENS = 128
DEFAULT_STALL_TIMEOUT_SEC = 25.0
DEFAULT_STALL_CHECK_INTERVAL_SEC = 5.0
SENTENCE_ENDINGS = {".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026"}
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]")


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _split_sentences(text: str) -> list[str]:
    segments: list[str] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in SENTENCE_ENDINGS:
            part = text[start : idx + 1].strip()
            if part:
                segments.append(part)
            start = idx + 1
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
        normalized = _normalize_text(segment)
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
            subtitle_cfg.get("max_repeat_sentences"), 2
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
        if session.translation_task and not session.translation_task.done():
            session.translation_task.cancel()
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
    text: str,
    language: Optional[str],
    is_final: bool,
    seq: int,
) -> None:
    try:
        translated = await _translate_text(session, text, language, websocket)
    except asyncio.CancelledError:
        return
    if not translated:
        if seq == _current_seq(session):
            session.seq = seq
            session.pending_seq = None
            session.utterance_text = ""
            session.utterance_language = None
        return
    if seq != _current_seq(session):
        return
    await _send_subtitle(
        websocket,
        text,
        translated,
        language,
        is_final,
        seq,
    )
    if seq == _current_seq(session):
        session.seq = seq
        session.pending_seq = None
        session.utterance_text = ""
        session.utterance_language = None


async def _start_translation(
    websocket: WebSocket,
    session: TranslationSession,
    text: str,
    language: Optional[str],
    is_final: bool,
    seq: int,
) -> None:
    if session.translation_task and not session.translation_task.done():
        session.translation_task.cancel()
    session.translation_task = asyncio.create_task(
        _translate_and_send(websocket, session, text, language, is_final, seq)
    )


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
    await _start_translation(
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


async def _apply_stt_update(
    stt_session: SttSession,
    translation_session: TranslationSession,
    stt_update: Optional[Dict[str, Any]],
    websocket: WebSocket,
) -> None:
    if not stt_update or not isinstance(stt_update, dict):
        return
    current_cfg = stt_session.cfg
    merged_cfg = _merge_config(current_cfg, stt_update)
    if not _should_refresh_engine(current_cfg, merged_cfg):
        return
    try:
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
            await _schedule_translation_debounce(websocket, session)
        else:
            if not session.translate_partials:
                continue
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
            subtitle_cfg.get("max_repeat_sentences"), 2
        ),
    }
    app.state.translator_cache = LRUCache(translator_cache_size)
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
    engine_kwargs = _build_engine_kwargs(stt_cfg)
    engine_kwargs["pcm_input"] = bool(stt_cfg.get("pcm_input", False))
    stt_engine = TranscriptionEngine(**engine_kwargs)
    translator = await _resolve_translator(
        app.state.translator_cache, app.state.translation_cfg
    )
    audio_processor = AudioProcessor(
        transcription_engine=stt_engine,
    )
    stt_session = SttSession(stt_cfg, stt_engine, audio_processor)
    session = TranslationSession(
        app.state.translation_cfg,
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

    async def _reset_stt_engine(reason: str) -> None:
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
                if not results_task.done():
                    results_task.cancel()
                    try:
                        await results_task
                    except asyncio.CancelledError:
                        pass
                await audio_processor.cleanup()
                engine_kwargs = _build_engine_kwargs(stt_session.cfg)
                engine_kwargs["pcm_input"] = bool(stt_session.cfg.get("pcm_input", False))
                stt_engine = TranscriptionEngine(**engine_kwargs)
                audio_processor = AudioProcessor(transcription_engine=stt_engine)
                stt_session.engine = stt_engine
                stt_session.audio_processor = audio_processor
                stt_session.last_audio_ts = time.monotonic()
                stt_session.last_result_ts = time.monotonic()
                stt_session.result_seen = False
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
            if not stt_session.result_seen:
                continue
            now = time.monotonic()
            if now - stt_session.last_audio_ts > stall_timeout:
                continue
            if now - stt_session.last_result_ts > stall_timeout:
                await _reset_stt_engine("stall")

    watchdog_task = asyncio.create_task(_stt_watchdog())

    try:
        await _safe_send(websocket, {"type": "status", "message": "connected"})
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                stt_session.last_audio_ts = time.monotonic()
                await audio_processor.process_audio(message["bytes"])
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
