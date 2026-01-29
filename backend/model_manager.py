from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional

MODEL_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

WHISPER_MODEL_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

ProgressCallback = Callable[[int, Optional[int]], None]
CancelCallback = Callable[[], bool]


def model_filename(model_size: str) -> str:
    return f"{model_size}.pt"


def resolve_model_target(
    model_name: str,
    model_cache_dir: Optional[str],
    model_path: Optional[str] = None,
) -> Optional[Path]:
    if model_path:
        return Path(model_path)
    if not model_cache_dir:
        return None
    return Path(model_cache_dir) / model_filename(model_name)


def resolve_stt_model_target(stt_cfg: Dict[str, Any]) -> Optional[Path]:
    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return None
    model_path = stt_cfg.get("model_path")
    model_cache_dir = stt_cfg.get("model_cache_dir")
    return resolve_model_target(model_name, model_cache_dir, model_path)


def download_model(
    url: str,
    target: Path,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
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
                if cancel_cb and cancel_cb():
                    raise RuntimeError("Download cancelled")
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


async def _maybe_call(callback: Optional[Callable[[str], object]], message: str) -> None:
    if not callback:
        return
    result = callback(message)
    if asyncio.iscoroutine(result):
        await result


async def ensure_model_available(
    stt_cfg: Dict[str, Any],
    *,
    notify_cb: Optional[Callable[[str], object]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    download_lock: Optional[asyncio.Lock] = None,
) -> Dict[str, Any]:
    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return stt_cfg
    url = WHISPER_MODEL_URLS.get(model_name)
    if not url:
        return stt_cfg
    target = resolve_stt_model_target(stt_cfg)
    if target is None:
        return stt_cfg
    if target.exists():
        updated = dict(stt_cfg)
        updated["model_path"] = str(target)
        return updated
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = download_lock or asyncio.Lock()
    async with lock:
        if target.exists():
            updated = dict(stt_cfg)
            updated["model_path"] = str(target)
            return updated
        await _maybe_call(notify_cb, f"Downloading Whisper model: {model_name}")
        await asyncio.to_thread(download_model, url, target, progress_cb)
        await _maybe_call(notify_cb, f"Model downloaded: {model_name}")
    updated = dict(stt_cfg)
    updated["model_path"] = str(target)
    return updated
