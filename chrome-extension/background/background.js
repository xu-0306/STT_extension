const OFFSCREEN_URL = chrome.runtime.getURL("offscreen/offscreen.html");
let captureActive = false;
let lastStatus = "Idle";
let activeTabId = null; // 追蹤正在錄音的 tab ID

chrome.storage.local.get(["captureActive", "lastStatus", "activeTabId"], (result) => {
  captureActive = Boolean(result.captureActive);
  lastStatus = result.lastStatus || "Idle";
  if (Number.isInteger(result.activeTabId)) {
    activeTabId = result.activeTabId;
  }
});

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

function updateState(active, status) {
  if (typeof active === "boolean") {
    captureActive = active;
  }
  if (status) {
    lastStatus = status;
  }
  chrome.storage.local.set({ captureActive, lastStatus });
}

function setActiveTabId(tabId) {
  activeTabId = Number.isInteger(tabId) ? tabId : null;
  chrome.storage.local.set({ activeTabId });
}

function loadStoredActiveTabId() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["activeTabId"], (result) => {
      if (chrome.runtime.lastError) {
        resolve(null);
        return;
      }
      resolve(Number.isInteger(result.activeTabId) ? result.activeTabId : null);
    });
  });
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

async function startCapture(wsUrl, settings) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    throw new Error("No active tab");
  }
  setActiveTabId(tab.id); // 儲存 tab ID 用於發送字幕
  try {
    await ensureContentScript(activeTabId);
    await ensureOffscreenDocument();
    await sendOffscreenMessage({ type: "offscreen-stop" });
    await new Promise((resolve) => setTimeout(resolve, 150));
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
    const response = await sendOffscreenMessage({
      type: "offscreen-start",
      streamId,
      wsUrl,
      settings,
    });
    if (!response.ok) {
      throw new Error(response.error || "start failed");
    }
    updateState(true, "Capturing");
  } catch (err) {
    const message = err?.message || "start failed";
    setActiveTabId(null);
    updateState(false, message);
    throw new Error(message);
  }
}

async function stopCapture() {
  await ensureOffscreenDocument();
  await sendOffscreenMessage({ type: "offscreen-stop" });
  let targetTabId = activeTabId;
  if (!Number.isInteger(targetTabId)) {
    targetTabId = await loadStoredActiveTabId();
  }
  if (Number.isInteger(targetTabId)) {
    await sendSubtitleToTab(targetTabId, { type: "subtitle-remove" });
  }
  setActiveTabId(null);
  updateState(false, "Stopped");
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

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "popup-start") {
    startCapture(message.wsUrl, message.settings)
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "popup-stop") {
    stopCapture()
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "popup-status") {
    chrome.offscreen.hasDocument().then((hasDocument) => {
      if (!hasDocument) {
        updateState(false, "Idle");
        sendResponse({ ok: true, capturing: false, status: lastStatus });
        return;
      }
      sendOffscreenMessage({ type: "offscreen-status" }).then((response) => {
        if (response && response.ok) {
          updateState(response.running, response.status || lastStatus);
          sendResponse({
            ok: true,
            capturing: response.running,
            status: response.status || lastStatus,
          });
          return;
        }
        sendResponse({ ok: true, capturing: captureActive, status: lastStatus });
      });
    });
    return true;
  }

  if (message.type === "offscreen-state") {
    updateState(message.running, message.status);
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
    console.log("[background] Received offscreen-subtitle:", payload?.type, "activeTabId:", activeTabId);
    if (activeTabId && payload) {
      if (payload.type === "subtitle") {
        console.log("[background] Forwarding subtitle to tab", activeTabId, ":", payload.original?.slice(0, 30));
        sendSubtitleToTab(activeTabId, {
          type: "subtitle",
          original: payload.original,
          translated: payload.translated,
        });
      } else if (payload.type === "status") {
        sendSubtitleToTab(activeTabId, {
          type: "subtitle-status",
          message: payload.message,
        });
      } else if (payload.type === "error") {
        sendSubtitleToTab(activeTabId, {
          type: "subtitle-status",
          message: `Error: ${payload.message}`,
        });
      }
    } else {
      console.warn("[background] Cannot forward subtitle: activeTabId=", activeTabId, "payload=", !!payload);
    }
    sendResponse({ ok: true });
    return true;
  }

  return false;
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (!captureActive || tabId !== activeTabId) return;
  if (changeInfo.status === "complete") {
    ensureContentScript(tabId);
  }
});
