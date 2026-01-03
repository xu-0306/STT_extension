# STT Subtitle Capture

Capture tab audio in Chrome and stream it to a local STT server for live subtitles
and translation. The STT implementation is based on WhisperLiveKit:
https://github.com/QuentinFuxa/WhisperLiveKit

## Features
- Chrome extension (MV3) that captures tab audio (WebM/Opus) and streams it over WebSocket.
- FastAPI backend that runs WhisperLiveKit for live transcription.
- Optional translation via NLLB, Ollama, OpenAI, or no translation.
- On-page subtitle overlay with adjustable position, size, and opacity.

## Project Structure
- `backend/`: FastAPI server, STT, translation, and config.
- `chrome-extension/`: Chrome extension (popup, background, content, offscreen).

## Requirements
- Python 3.10+
- Chrome or Edge (for loading the extension)
- PyTorch installed for your CUDA or CPU environment

## Quick Start
### 1) Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately for your CUDA version (see `backend/requirements.txt`).

Start the server:
```bash
python server.py
```

By default the server listens on `ws://127.0.0.1:8765/asr`.
Only local connections are allowed.

### 2) Chrome Extension
1. Open `chrome://extensions` in Chrome.
2. Enable Developer mode.
3. Click "Load unpacked" and select `chrome-extension/`.
4. Open a tab with audio.
5. Click the extension icon, set the WebSocket URL if needed, then click "Start Capture".

## Configuration
Backend config lives in `backend/config.yaml`.

Key options:
- `server.host` / `server.port`: server bind address and port
- `stt.model`: Whisper model size (e.g. `tiny`, `small`, `medium`, `large-v3`)
- `translation.engine`: `nllb`, `ollama`, `openai`, or `noop`

The extension also lets you select translation models and target language
from the Options page (click "Setting" in the popup).

## Notes
- The extension streams WebM/Opus audio via MediaRecorder.
- For Ollama translation, ensure your Ollama server is running.
- For OpenAI translation, set your API key in the extension Options page.

## Credits
- WhisperLiveKit (STT engine): https://github.com/QuentinFuxa/WhisperLiveKit
