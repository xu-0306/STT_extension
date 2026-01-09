const OFFSCREEN_URL = chrome.runtime.getURL("offscreen/offscreen.html");
const tabStatus = new Map();

async function ensureContentScript(tabId) {
  if (!tabId) return false;
  try {
    await chrome.scripting.insertCSS({
      target: { tabId },
      files: ["content/content.css"],
    });
  } catch (err) {
    console.warn("[background] Failed to inject CSS:", err.message);
  }
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content/content.js"],
    });
    return true;
  } catch (err) {
    console.warn("[background] Failed to inject content script:", err.message);
    return false;
  }
}

async function resolveTabId(tabId) {
  if (Number.isInteger(tabId)) {
    return tabId;
  }
  if (typeof tabId === "string" && tabId.trim()) {
    const parsed = Number.parseInt(tabId, 10);
    if (Number.isInteger(parsed)) {
      return parsed;
    }
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab?.id ?? null;
}

function sendOffscreenMessage(payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(payload, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { ok: false, error: "no response" });
    });
  });
}

async function ensureOffscreenDocument() {
  const hasDocument = await chrome.offscreen.hasDocument();
  if (hasDocument) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["AUDIO_PLAYBACK"],
    justification: "Capture tab audio and stream PCM to a local STT server.",
  });
}

async function startCapture(tabId, wsUrl, settings) {
  const targetTabId = await resolveTabId(tabId);
  if (!targetTabId) {
    throw new Error("No active tab");
  }
  try {
    await ensureContentScript(targetTabId);
    await ensureOffscreenDocument();
    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId,
    });
    const response = await sendOffscreenMessage({
      type: "offscreen-start",
      tabId: targetTabId,
      streamId,
      wsUrl,
      settings,
    });
    if (!response.ok) {
      throw new Error(response.error || "start failed");
    }
  } catch (err) {
    throw new Error(err?.message || "start failed");
  }
}

async function stopCapture(tabId) {
  const targetTabId = await resolveTabId(tabId);
  if (!targetTabId) return;
  await ensureOffscreenDocument();
  await sendOffscreenMessage({ type: "offscreen-stop", tabId: targetTabId });
  await sendSubtitleToTab(targetTabId, { type: "subtitle-remove" });
}

// 發送字幕到 content script
async function sendSubtitleToTab(tabId, payload) {
  if (!tabId) return;
  try {
    await chrome.tabs.sendMessage(tabId, payload);
  } catch (err) {
    // Tab 可能已關閉或沒有 content script
    console.warn("[background] Failed to send subtitle to tab:", err.message);
    const injected = await ensureContentScript(tabId);
    if (!injected) return;
    try {
      await chrome.tabs.sendMessage(tabId, payload);
    } catch (retryErr) {
      console.warn("[background] Retry send failed:", retryErr.message);
    }
  }
}

function buildFallbackStatus(requestedTabId) {
  const status = tabStatus.get(requestedTabId) || { running: false, status: "Idle" };
  const anyRunning = Array.from(tabStatus.values()).some((item) => item.running);
  return {
    ok: true,
    capturing: Boolean(status.running),
    status: status.status || (status.running ? "Capturing" : "Idle"),
    globalCapturing: anyRunning,
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "popup-start") {
    startCapture(message.tabId, message.wsUrl, message.settings)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "popup-stop") {
    stopCapture(message.tabId)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "popup-status") {
    const requestedTabId = Number.isInteger(message.tabId) ? message.tabId : null;
    chrome.offscreen.hasDocument().then((hasDocument) => {
      if (!hasDocument) {
        sendResponse({
          ok: true,
          capturing: false,
          status: "Idle",
          globalCapturing: false,
        });
        return;
      }
      sendOffscreenMessage({ type: "offscreen-status", tabId: requestedTabId })
        .then((response) => {
          if (response && response.ok) {
            sendResponse({
              ok: true,
              capturing: Boolean(response.running),
              status: response.status || (response.running ? "Capturing" : "Idle"),
              globalCapturing: Boolean(response.anyRunning),
            });
            return;
          }
          sendResponse(buildFallbackStatus(requestedTabId));
        })
        .catch(() => {
          sendResponse(buildFallbackStatus(requestedTabId));
        });
    });
    return true;
  }

  if (message.type === "offscreen-state") {
    if (Number.isInteger(message.tabId)) {
      tabStatus.set(message.tabId, {
        running: Boolean(message.running),
        status: message.status || (message.running ? "Capturing" : "Idle"),
      });
    }
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === "offscreen-keepalive") {
    sendResponse({ ok: true });
    return true;
  }

  // 處理來自 offscreen 的字幕訊息，轉發給 content script
  if (message.type === "offscreen-subtitle") {
    const payload = message.payload;
    const tabId = message.tabId;
    if (tabId && payload) {
      if (payload.type === "subtitle") {
        sendSubtitleToTab(tabId, {
          type: "subtitle",
          original: payload.original,
          translated: payload.translated,
          seq: payload.seq,
        });
      } else if (payload.type === "status") {
        sendSubtitleToTab(tabId, {
          type: "subtitle-status",
          message: payload.message,
        });
      } else if (payload.type === "error") {
        sendSubtitleToTab(tabId, {
          type: "subtitle-status",
          message: `Error: ${payload.message}`,
        });
      }
    }
    sendResponse({ ok: true });
    return true;
  }

  return false;
});
