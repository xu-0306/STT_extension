import asyncio
import ipaddress
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import AudioProcessor, TranscriptionEngine

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
        "max_context_tokens": simul_cfg.get("max_context_tokens"),
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


def _translation_cache_key(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _normalize_cache_size(value: object, default: int) -> int:
    try:
        size = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, size)


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
    ) -> None:
        self.default_lang = default_lang
        self.cfg = dict(cfg)
        self.translator = translator
        self.translate_partials = bool(self.cfg.get("partial", False))


async def _safe_send(websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    try:
        await websocket.send_text(json.dumps(payload))
        return True
    except (WebSocketDisconnect, RuntimeError, OSError):
        return False


def _should_refresh_engine(current: Dict[str, Any], updated: Dict[str, Any]) -> bool:
    return current != updated


async def _apply_stt_update(
    app: FastAPI,
    audio_processor: AudioProcessor,
    stt_update: Optional[Dict[str, Any]],
    websocket: WebSocket,
) -> None:
    if not stt_update or not isinstance(stt_update, dict):
        return
    current_cfg = app.state.stt_cfg
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
    app.state.stt_cfg = merged_cfg
    app.state.default_language = str(merged_cfg.get("language", "auto"))
    app.state.transcription_engine = new_engine
    audio_processor.transcription_engine = new_engine
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

        language = _guess_language(text, language or session.default_lang)
        if is_final:
            if text == last_final:
                continue
            last_final = text
        else:
            if not session.translate_partials:
                continue
            if text == last_partial:
                continue
            last_partial = text

        translated = ""
        try:
            translated = await asyncio.to_thread(session.translator.translate, text, language)
        except Exception as exc:
            await _safe_send(
                websocket,
                {"type": "error", "message": f"Translation error: {type(exc).__name__}"},
            )

        await _safe_send(
            websocket,
            {
                "type": "subtitle",
                "original": text,
                "translated": translated,
                "language": language,
                "timestamp": int(time.time() * 1000),
                "final": is_final,
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.server_config = _get_server_config(cfg)
    stt_cfg = cfg.get("stt", {})
    translation_cfg = cfg.get("translation", {})
    translator_cache_size = _normalize_cache_size(
        translation_cfg.get("translator_cache_size", 4),
        4,
    )

    engine_kwargs = _build_engine_kwargs(stt_cfg)
    engine_kwargs["pcm_input"] = bool(stt_cfg.get("pcm_input", False))
    app.state.default_language = str(stt_cfg.get("language", "auto"))
    app.state.stt_cfg = stt_cfg
    app.state.translation_cfg = translation_cfg
    app.state.translator_cache = LRUCache(translator_cache_size)
    app.state.active_ws = False
    app.state.active_ws_lock = asyncio.Lock()
    app.state.transcription_engine = TranscriptionEngine(**engine_kwargs)
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
    claimed = False
    async with app.state.active_ws_lock:
        if app.state.active_ws:
            await _safe_send(
                websocket,
                {"type": "error", "message": "Another session is already active"},
            )
            await websocket.close(code=1008)
            return
        app.state.active_ws = True
        claimed = True
    translator = await _resolve_translator(
        app.state.translator_cache, app.state.translation_cfg
    )
    session = TranslationSession(
        app.state.translation_cfg,
        app.state.default_language,
        translator,
    )
    audio_processor = AudioProcessor(
        transcription_engine=app.state.transcription_engine,
    )
    results_generator = await audio_processor.create_tasks()
    results_task = asyncio.create_task(
        _handle_results(websocket, results_generator, session)
    )

    try:
        await _safe_send(websocket, {"type": "status", "message": "connected"})
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
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
                            app,
                            audio_processor,
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
        if not results_task.done():
            results_task.cancel()
            try:
                await results_task
            except asyncio.CancelledError:
                pass
        await audio_processor.cleanup()
        if claimed:
            async with app.state.active_ws_lock:
                app.state.active_ws = False


def main() -> None:
    import uvicorn

    cfg = load_config()
    server_cfg = _get_server_config(cfg)
    uvicorn.run(
        app,
        host=server_cfg["host"],
        port=server_cfg["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
