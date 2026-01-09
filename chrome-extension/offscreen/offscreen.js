const sessions = new Map();

const KEEP_ALIVE_INTERVAL_MS = 15000;
const RECORDER_CHUNK_MS = 100;
const MAX_PENDING_BYTES = 8 * 1024 * 1024;
const MAX_BUFFERED_BYTES = 2 * 1024 * 1024;
const PENDING_FLUSH_INTERVAL_MS = 500;

function normalizeTabId(value) {
  if (Number.isInteger(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isInteger(parsed)) {
      return parsed;
    }
  }
  return null;
}

function createSession(tabId, wsUrl, settings) {
  return {
    tabId,
    wsUrl,
    settings,
    mediaStream: null,
    mediaRecorder: null,
    ws: null,
    running: false,
    keepAliveTimer: null,
    wsPingTimer: null,
    flushTimer: null,
    pendingChunks: [],
    pendingBytes: 0,
    chunkChain: Promise.resolve(),
    audioContext: null,
    sourceNode: null,
    monitorGain: null,
    closing: false,
    disconnectHandled: false,
  };
}

function isSessionActive(session) {
  return sessions.get(session.tabId) === session;
}

function notifyBackground(tabId, status, runningOverride) {
  chrome.runtime.sendMessage({
    type: "offscreen-state",
    tabId,
    status,
    running: typeof runningOverride === "boolean" ? runningOverride : false,
  });
}

function startKeepAlive(session) {
  stopKeepAlive(session);
  session.keepAliveTimer = setInterval(() => {
    chrome.runtime.sendMessage({ type: "offscreen-keepalive" });
  }, KEEP_ALIVE_INTERVAL_MS);
  session.wsPingTimer = setInterval(() => {
    if (session.ws && session.ws.readyState === WebSocket.OPEN) {
      session.ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
    }
  }, KEEP_ALIVE_INTERVAL_MS);
  session.flushTimer = setInterval(() => {
    flushPendingChunks(session);
  }, PENDING_FLUSH_INTERVAL_MS);
}

function stopKeepAlive(session) {
  if (session.keepAliveTimer) {
    clearInterval(session.keepAliveTimer);
    session.keepAliveTimer = null;
  }
  if (session.wsPingTimer) {
    clearInterval(session.wsPingTimer);
    session.wsPingTimer = null;
  }
  if (session.flushTimer) {
    clearInterval(session.flushTimer);
    session.flushTimer = null;
  }
}

function canSendNow(session) {
  return (
    session.ws &&
    session.ws.readyState === WebSocket.OPEN &&
    session.ws.bufferedAmount < MAX_BUFFERED_BYTES
  );
}

function enqueueChunk(session, buffer) {
  if (!buffer || buffer.byteLength === 0) {
    return;
  }
  if (session.pendingChunks.length === 0 && canSendNow(session)) {
    session.ws.send(buffer);
    return;
  }
  session.pendingChunks.push(buffer);
  session.pendingBytes += buffer.byteLength;
  while (session.pendingBytes > MAX_PENDING_BYTES && session.pendingChunks.length > 0) {
    const dropped = session.pendingChunks.shift();
    if (dropped) {
      session.pendingBytes -= dropped.byteLength;
    }
  }
  if (session.ws && session.ws.readyState === WebSocket.OPEN) {
    flushPendingChunks(session);
  }
}

function flushPendingChunks(session) {
  if (!session.ws || session.ws.readyState !== WebSocket.OPEN || session.pendingChunks.length === 0) {
    return;
  }
  while (session.pendingChunks.length > 0 && session.ws.bufferedAmount < MAX_BUFFERED_BYTES) {
    const chunk = session.pendingChunks.shift();
    if (!chunk) {
      continue;
    }
    session.pendingBytes -= chunk.byteLength;
    session.ws.send(chunk);
  }
  if (session.pendingChunks.length === 0) {
    session.pendingBytes = 0;
  } else if (session.pendingBytes < 0) {
    session.pendingBytes = 0;
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

async function startCapture(tabId, streamId, wsUrl, settings) {
  const normalizedTabId = normalizeTabId(tabId);
  if (!normalizedTabId) {
    throw new Error("Invalid tab id");
  }
  if (sessions.has(normalizedTabId)) {
    await stopCapture(normalizedTabId);
  }

  const session = createSession(normalizedTabId, wsUrl, settings);
  sessions.set(normalizedTabId, session);

  session.running = true;
  session.closing = false;
  session.disconnectHandled = false;
  notifyBackground(tabId, "Starting...", true);
  startKeepAlive(session);
  session.pendingChunks = [];
  session.pendingBytes = 0;
  session.chunkChain = Promise.resolve();

  try {
    session.ws = new WebSocket(wsUrl);
    session.ws.binaryType = "arraybuffer";
    session.ws.addEventListener("open", () => {
      if (!isSessionActive(session)) return;
      notifyBackground(tabId, "Connected", true);
      if (settings && (settings.translation || settings.stt)) {
        const payload = { type: "config" };
        if (settings.translation) {
          payload.translation = settings.translation;
        }
        if (settings.stt) {
          payload.stt = settings.stt;
        }
        session.ws.send(JSON.stringify(payload));
      }
      flushPendingChunks(session);
    });
    session.ws.addEventListener("message", (event) => {
      if (!isSessionActive(session)) return;
      if (typeof event.data !== "string") {
        return;
      }
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "subtitle" || payload.type === "status" || payload.type === "error") {
          chrome.runtime.sendMessage(
            { type: "offscreen-subtitle", tabId: session.tabId, payload },
            () => {}
          );
        }
      } catch (err) {
        console.error("[offscreen] JSON parse error:", err.message);
      }
    });
    session.ws.addEventListener("close", (event) => {
      if (!isSessionActive(session)) return;
      handleWsDisconnect(session, `Disconnected (${event.code || 0})`);
    });
    session.ws.addEventListener("error", () => {
      if (!isSessionActive(session)) return;
      handleWsDisconnect(session, "WebSocket error");
    });

    session.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
    });

    try {
      // Route captured audio to output so the tab doesn't go silent during capture.
      session.audioContext = new AudioContext();
      await session.audioContext.resume();
      session.audioContext.addEventListener("statechange", () => {
        if (session.running && session.audioContext && session.audioContext.state === "suspended") {
          session.audioContext.resume();
        }
      });
      session.sourceNode = session.audioContext.createMediaStreamSource(session.mediaStream);
      session.monitorGain = session.audioContext.createGain();
      session.monitorGain.gain.value = 1;
      session.sourceNode.connect(session.monitorGain);
      session.monitorGain.connect(session.audioContext.destination);
    } catch (err) {
      console.warn("[offscreen] audio monitor init error:", err?.message || err);
    }

    const mimeType = pickMimeType();
    const options = mimeType ? { mimeType } : undefined;
    session.mediaRecorder = new MediaRecorder(session.mediaStream, options);
    session.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (!isSessionActive(session)) return;
      const blob = event.data;
      if (!blob || blob.size === 0) {
        return;
      }
      session.chunkChain = session.chunkChain
        .then(async () => {
          const buffer = await blob.arrayBuffer();
          enqueueChunk(session, buffer);
        })
        .catch((err) => {
          console.warn("[offscreen] chunk encode error:", err?.message || err);
        });
    });
    session.mediaRecorder.start(RECORDER_CHUNK_MS);
  } catch (err) {
    session.running = false;
    session.closing = true;
    session.disconnectHandled = true;
    notifyBackground(tabId, err?.message || "Capture start failed", false);
    await cleanupCaptureResources(session, { closeWebSocket: true });
    session.closing = false;
    if (isSessionActive(session)) {
      sessions.delete(normalizedTabId);
    }
    throw err;
  }
}

async function cleanupCaptureResources(session, options = {}) {
  const closeWebSocket = options.closeWebSocket !== false;

  if (session.mediaRecorder) {
    try {
      const recorder = session.mediaRecorder;
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
        session.chunkChain,
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
    } catch (err) {
      console.warn("[offscreen] MediaRecorder stop error:", err.message);
    }
    session.mediaRecorder = null;
  }

  if (session.mediaStream) {
    session.mediaStream.getTracks().forEach((track) => track.stop());
    session.mediaStream = null;
  }

  if (session.monitorGain) {
    session.monitorGain.disconnect();
    session.monitorGain = null;
  }

  if (session.sourceNode) {
    session.sourceNode.disconnect();
    session.sourceNode = null;
  }

  if (session.audioContext) {
    try {
      await session.audioContext.close();
    } catch (err) {
      console.warn("[offscreen] AudioContext close error:", err?.message || err);
    }
    session.audioContext = null;
  }

  if (closeWebSocket && session.ws) {
    session.ws.close();
    session.ws = null;
  }
  if (!closeWebSocket) {
    session.ws = null;
  }
  stopKeepAlive(session);
}

function handleWsDisconnect(session, message) {
  if (session.closing || session.disconnectHandled) return;
  session.disconnectHandled = true;
  session.running = false;
  notifyBackground(session.tabId, message, false);
  if (session.ws && session.ws.readyState === WebSocket.OPEN) {
    try {
      session.ws.close();
    } catch (err) {
      // Ignore close errors.
    }
  }
  session.ws = null;
  cleanupCaptureResources(session, { closeWebSocket: false })
    .catch((err) => {
      console.warn("[offscreen] cleanup error:", err?.message || err);
    })
    .finally(() => {
      if (isSessionActive(session)) {
        sessions.delete(session.tabId);
      }
    });
}

async function stopCapture(tabId) {
  const normalizedTabId = normalizeTabId(tabId);
  if (!normalizedTabId) {
    throw new Error("Invalid tab id");
  }
  const session = sessions.get(normalizedTabId);
  if (!session) return;
  session.running = false;
  session.closing = true;
  session.disconnectHandled = true;
  notifyBackground(tabId, "Stopping...", false);
  await cleanupCaptureResources(session, { closeWebSocket: true });
  session.closing = false;
  if (isSessionActive(session)) {
    sessions.delete(normalizedTabId);
  }
  notifyBackground(tabId, "Stopped", false);
}

async function stopAllCaptures() {
  const tabIds = Array.from(sessions.keys());
  for (const tabId of tabIds) {
    await stopCapture(tabId);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "offscreen-start") {
    const tabId = message.tabId;
    startCapture(tabId, message.streamId, message.wsUrl, message.settings)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "offscreen-status") {
    const requestedTabId = normalizeTabId(message.tabId);
    const session = requestedTabId !== null ? sessions.get(requestedTabId) : null;
    const running = Boolean(session?.running);
    const anyRunning = Array.from(sessions.values()).some((item) => item.running);
    sendResponse({
      ok: true,
      running,
      status: running ? "Capturing" : "Idle",
      wsReadyState: session?.ws ? session.ws.readyState : null,
      anyRunning,
    });
    return true;
  }

  if (message.type === "offscreen-stop") {
    const tabId = message.tabId;
    const stopPromise = Number.isInteger(tabId) ? stopCapture(tabId) : stopAllCaptures();
    stopPromise
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  return false;
});
