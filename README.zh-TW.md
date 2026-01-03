# STT Subtitle Capture

在 Chrome 擴充功能中擷取分頁音訊，串流到本機 STT 伺服器以產生即時字幕與翻譯。
本專案的 STT 基於 WhisperLiveKit 實現：
https://github.com/QuentinFuxa/WhisperLiveKit

## 功能
- Chrome 擴充功能（MV3）擷取分頁音訊（WebM/Opus）並透過 WebSocket 串流。
- FastAPI 後端使用 WhisperLiveKit 進行即時語音轉文字。
- 翻譯可選 NLLB、Ollama、OpenAI 或關閉翻譯。
- 網頁字幕疊加層可調整位置、大小與透明度。

## 專案結構
- `backend/`: FastAPI 伺服器、STT、翻譯與設定。
- `chrome-extension/`: Chrome 擴充（popup、background、content、offscreen）。

## 需求
- Python 3.10+
- Chrome 或 Edge（用於載入擴充）
- PyTorch（請依照 CUDA 或 CPU 環境自行安裝）

## 快速開始
### 1) 後端
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

請另外安裝對應 CUDA 版本的 PyTorch（參考 `backend/requirements.txt` 註解）。

啟動伺服器：
```bash
python server.py
```

預設 WebSocket 位址為 `ws://127.0.0.1:8765/asr`，且僅允許本機連線。

### 2) Chrome 擴充
1. 開啟 `chrome://extensions`
2. 啟用開發人員模式
3. 點選「載入未封裝項目」，選擇 `chrome-extension/`
4. 開啟有播放音訊的分頁
5. 點擊擴充圖示，視需要修改 WebSocket URL，點擊「Start Capture」

## 設定
後端設定位於 `backend/config.yaml`。

常用設定：
- `server.host` / `server.port`: 伺服器綁定位址與連接埠
- `stt.model`: Whisper 模型尺寸（如 `tiny`、`small`、`medium`、`large-v3`）
- `translation.engine`: `nllb`、`ollama`、`openai` 或 `noop`

擴充功能也可在 Options 頁面選擇翻譯模型與目標語言（點擊 popup 的 Setting）。

## 備註
- 擴充功能透過 MediaRecorder 串流 WebM/Opus 音訊。
- 使用 Ollama 翻譯時請確認 Ollama 服務已啟動。
- 使用 OpenAI 翻譯時請在 Options 頁面設定 API key。

## Credits
- WhisperLiveKit（STT 引擎）：https://github.com/QuentinFuxa/WhisperLiveKit
