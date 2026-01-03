let mediaStream = null;
let mediaRecorder = null;
let ws = null;
let running = false;
let keepAliveTimer = null;
let wsPingTimer = null;
let flushTimer = null;
let pendingChunks = [];
let pendingBytes = 0;
let chunkChain = Promise.resolve();
let audioContext = null;
let sourceNode = null;
let monitorGain = null;
let closing = false;
let disconnectHandled = false;

const KEEP_ALIVE_INTERVAL_MS = 15000;
const RECORDER_CHUNK_MS = 100;
const MAX_PENDING_BYTES = 8 * 1024 * 1024;
const MAX_BUFFERED_BYTES = 2 * 1024 * 1024;
const PENDING_FLUSH_INTERVAL_MS = 500;

function notifyBackground(status, runningOverride) {
  chrome.runtime.sendMessage({
    type: "offscreen-state",
    status,
    running: typeof runningOverride === "boolean" ? runningOverride : running,
  });
}

function startKeepAlive() {
  stopKeepAlive();
  keepAliveTimer = setInterval(() => {
    chrome.runtime.sendMessage({ type: "offscreen-keepalive" });
  }, KEEP_ALIVE_INTERVAL_MS);
  wsPingTimer = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
    }
  }, KEEP_ALIVE_INTERVAL_MS);
  flushTimer = setInterval(() => {
    flushPendingChunks();
  }, PENDING_FLUSH_INTERVAL_MS);
}

function stopKeepAlive() {
  if (keepAliveTimer) {
    clearInterval(keepAliveTimer);
    keepAliveTimer = null;
  }
  if (wsPingTimer) {
    clearInterval(wsPingTimer);
    wsPingTimer = null;
  }
  if (flushTimer) {
    clearInterval(flushTimer);
    flushTimer = null;
  }
}

function canSendNow() {
  return (
    ws &&
    ws.readyState === WebSocket.OPEN &&
    ws.bufferedAmount < MAX_BUFFERED_BYTES
  );
}

function enqueueChunk(buffer) {
  if (!buffer || buffer.byteLength === 0) {
    return;
  }
  if (pendingChunks.length === 0 && canSendNow()) {
    ws.send(buffer);
    return;
  }
  pendingChunks.push(buffer);
  pendingBytes += buffer.byteLength;
  while (pendingBytes > MAX_PENDING_BYTES && pendingChunks.length > 0) {
    const dropped = pendingChunks.shift();
    if (dropped) {
      pendingBytes -= dropped.byteLength;
    }
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    flushPendingChunks();
  }
}

function flushPendingChunks() {
  if (!ws || ws.readyState !== WebSocket.OPEN || pendingChunks.length === 0) {
    return;
  }
  while (pendingChunks.length > 0 && ws.bufferedAmount < MAX_BUFFERED_BYTES) {
    const chunk = pendingChunks.shift();
    if (!chunk) {
      continue;
    }
    pendingBytes -= chunk.byteLength;
    ws.send(chunk);
  }
  if (pendingChunks.length === 0) {
    pendingBytes = 0;
  } else if (pendingBytes < 0) {
    pendingBytes = 0;
  }
}

function pickMimeType() {
  const preferred = "audio/webm;codecs=opus";
  if (MediaRecorder.isTypeSupported(preferred)) {
    return preferred;
  }
  if (MediaRecorder.isTypeSupported("audio/webm")) {
    return "audio/webm";
  }
  return "";
}

async function startCapture(streamId, wsUrl, settings) {
  if (running) {
    await stopCapture();
  }
  running = true;
  closing = false;
  disconnectHandled = false;
  notifyBackground("Starting...", true);
  startKeepAlive();
  pendingChunks = [];
  pendingBytes = 0;
  chunkChain = Promise.resolve();

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  ws.addEventListener("open", () => {
    notifyBackground("Connected", true);
    if (settings && (settings.translation || settings.stt)) {
      const payload = { type: "config" };
      if (settings.translation) {
        payload.translation = settings.translation;
      }
      if (settings.stt) {
        payload.stt = settings.stt;
      }
      ws.send(JSON.stringify(payload));
    }
    flushPendingChunks();
  });
  ws.addEventListener("message", (event) => {
    if (typeof event.data !== "string") {
      return;
    }
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "subtitle" || payload.type === "status" || payload.type === "error") {
        chrome.runtime.sendMessage(
          { type: "offscreen-subtitle", payload },
          () => {}
        );
      }
    } catch (err) {
      console.error("[offscreen] JSON parse error:", err.message);
    }
  });
  ws.addEventListener("close", (event) => {
    handleWsDisconnect(`Disconnected (${event.code || 0})`);
  });
  ws.addEventListener("error", () => {
    handleWsDisconnect("WebSocket error");
  });

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId,
      },
    },
  });

  try {
    // Route captured audio to output so the tab doesn't go silent during capture.
    audioContext = new AudioContext();
    await audioContext.resume();
    audioContext.addEventListener("statechange", () => {
      if (running && audioContext && audioContext.state === "suspended") {
        audioContext.resume();
      }
    });
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    monitorGain = audioContext.createGain();
    monitorGain.gain.value = 1;
    sourceNode.connect(monitorGain);
    monitorGain.connect(audioContext.destination);
  } catch (err) {
    console.warn("[offscreen] audio monitor init error:", err?.message || err);
  }

  const mimeType = pickMimeType();
  const options = mimeType ? { mimeType } : undefined;
  mediaRecorder = new MediaRecorder(mediaStream, options);
  mediaRecorder.addEventListener("dataavailable", (event) => {
    const blob = event.data;
    if (!blob || blob.size === 0) {
      return;
    }
    chunkChain = chunkChain
      .then(async () => {
        const buffer = await blob.arrayBuffer();
        enqueueChunk(buffer);
      })
      .catch((err) => {
        console.warn("[offscreen] chunk encode error:", err?.message || err);
      });
  });
  mediaRecorder.start(RECORDER_CHUNK_MS);
}

async function cleanupCaptureResources(options = {}) {
  const closeWebSocket = options.closeWebSocket !== false;

  if (mediaRecorder) {
    try {
      const recorder = mediaRecorder;
      const stopped = new Promise((resolve) => {
        recorder.addEventListener("stop", resolve, { once: true });
      });
      try {
        recorder.requestData();
      } catch (err) {
        // Ignore if recorder is inactive.
      }
      recorder.stop();
      await Promise.race([
        stopped,
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
      await Promise.race([
        chunkChain,
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
    } catch (err) {
      console.warn("[offscreen] MediaRecorder stop error:", err.message);
    }
    mediaRecorder = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (monitorGain) {
    monitorGain.disconnect();
    monitorGain = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (audioContext) {
    try {
      await audioContext.close();
    } catch (err) {
      console.warn("[offscreen] AudioContext close error:", err?.message || err);
    }
    audioContext = null;
  }

  if (closeWebSocket && ws) {
    ws.close();
    ws = null;
  }
  if (!closeWebSocket) {
    ws = null;
  }
  stopKeepAlive();
}

function handleWsDisconnect(message) {
  if (closing || disconnectHandled) return;
  disconnectHandled = true;
  running = false;
  notifyBackground(message, false);
  if (ws && ws.readyState === WebSocket.OPEN) {
    try {
      ws.close();
    } catch (err) {
      // Ignore close errors.
    }
  }
  ws = null;
  cleanupCaptureResources({ closeWebSocket: false }).catch((err) => {
    console.warn("[offscreen] cleanup error:", err?.message || err);
  });
}

async function stopCapture() {
  running = false;
  closing = true;
  disconnectHandled = true;
  notifyBackground("Stopping...", false);
  await cleanupCaptureResources({ closeWebSocket: true });
  closing = false;
  notifyBackground("Stopped", false);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "offscreen-start") {
    startCapture(message.streamId, message.wsUrl, message.settings)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "offscreen-status") {
    sendResponse({
      ok: true,
      running,
      status: running ? "Capturing" : "Idle",
      wsReadyState: ws ? ws.readyState : null,
    });
    return true;
  }

  if (message.type === "offscreen-stop") {
    stopCapture()
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  return false;
});
