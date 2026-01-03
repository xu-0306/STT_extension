(function () {
  "use strict";

  // 防止重複注入
  if (window.__sttSubtitleInjected) return;
  window.__sttSubtitleInjected = true;

  const HIDE_DELAY_MS = 0; // Keep the last subtitle visible until the next utterance starts.
  const SILENCE_CLEAR_MS = 2500; // Gap between updates to start a new utterance.
  const MAX_ORIGINAL_CHARS = 260;
  const MAX_TRANSLATED_CHARS = 260;
  const POSITION_STORAGE_KEY = "sttSubtitleOverlayPosition";
  const SIZE_STORAGE_KEY = "sttSubtitleOverlaySize";
  const DEFAULT_OVERLAY_WIDTH = 640;
  const DEFAULT_OVERLAY_HEIGHT = 160;
  const MIN_OVERLAY_WIDTH = 320;
  const MIN_OVERLAY_HEIGHT = 90;
  const RESIZE_HANDLE_SIZE = 16;
  const RESIZE_SAVE_DELAY_MS = 200;
  const DRAG_MARGIN = 12;
  const FONT_ORIGINAL_SCALE_KEY = "subtitleOriginalScale";
  const FONT_TRANSLATED_SCALE_KEY = "subtitleTranslatedScale";
  const LEGACY_FONT_SCALE_KEY = "subtitleFontScale";
  const DEFAULT_FONT_SCALE = 1;
  const MIN_FONT_SCALE = 0.7;
  const MAX_FONT_SCALE = 1.6;
  const BASE_ORIGINAL_FONT_SIZE = 18;
  const BASE_TRANSLATED_FONT_SIZE = 22;
  const OVERLAY_OPACITY_KEY = "subtitleOverlayOpacity";
  const DEFAULT_OVERLAY_OPACITY = 0.85;
  const MIN_OVERLAY_OPACITY = 0.35;
  const MAX_OVERLAY_OPACITY = 1;

  let overlayEl = null;
  let cardEl = null;
  let originalEl = null;
  let translatedEl = null;
  let statusEl = null;
  let hideTimer = null;
  let fullscreenListenerAttached = false;
  let dragHandlersAttached = false;
  let dragState = null;
  let overlayPosition = null;
  let positionLoaded = false;
  let overlaySize = null;
  let sizeLoaded = false;
  let resizeObserver = null;
  let resizeSaveTimer = null;
  let lastSubtitleTs = 0;
  let originalFontScale = DEFAULT_FONT_SCALE;
  let translatedFontScale = DEFAULT_FONT_SCALE;
  let fontScaleLoaded = false;
  let overlayOpacity = DEFAULT_OVERLAY_OPACITY;
  let overlayOpacityLoaded = false;

  function trimText(text, maxChars) {
    if (!text) return "";
    if (!maxChars || text.length <= maxChars) return text;
    const trimmed = text.slice(-maxChars);
    return trimmed.replace(/^[\s\.,;:!\?-]+/, "");
  }

  function clearSubtitleText() {
    if (originalEl) {
      originalEl.textContent = "";
      originalEl.style.display = "none";
    }
    if (translatedEl) {
      translatedEl.textContent = "";
      translatedEl.style.display = "none";
    }
  }

  function getOverlayHost() {
    return document.fullscreenElement || document.body;
  }

  function ensureOverlayHost() {
    if (!overlayEl) return;
    const host = getOverlayHost();
    if (host && overlayEl.parentNode !== host) {
      host.appendChild(overlayEl);
    }
  }

  function loadStoredPosition() {
    return new Promise((resolve) => {
      if (!chrome?.storage?.local) {
        resolve(null);
        return;
      }
      chrome.storage.local.get([POSITION_STORAGE_KEY], (result) => {
        if (chrome.runtime.lastError) {
          resolve(null);
          return;
        }
        const stored = result[POSITION_STORAGE_KEY];
        if (stored && typeof stored.x === "number" && typeof stored.y === "number") {
          resolve(stored);
          return;
        }
        resolve(null);
      });
    });
  }

  function loadStoredSize() {
    return new Promise((resolve) => {
      if (!chrome?.storage?.local) {
        resolve(null);
        return;
      }
      chrome.storage.local.get([SIZE_STORAGE_KEY], (result) => {
        if (chrome.runtime.lastError) {
          resolve(null);
          return;
        }
        const stored = result[SIZE_STORAGE_KEY];
        if (stored && typeof stored.width === "number" && typeof stored.height === "number") {
          resolve(stored);
          return;
        }
        resolve(null);
      });
    });
  }

  function saveStoredPosition(position) {
    if (!chrome?.storage?.local) return;
    chrome.storage.local.set({ [POSITION_STORAGE_KEY]: position });
  }

  function saveStoredSize(size) {
    if (!chrome?.storage?.local) return;
    chrome.storage.local.set({ [SIZE_STORAGE_KEY]: size });
  }

  function loadStoredFontScales() {
    return new Promise((resolve) => {
      if (!chrome?.storage?.local) {
        resolve(null);
        return;
      }
      chrome.storage.local.get(
        [FONT_ORIGINAL_SCALE_KEY, FONT_TRANSLATED_SCALE_KEY, LEGACY_FONT_SCALE_KEY],
        (result) => {
          if (chrome.runtime.lastError) {
            resolve(null);
            return;
          }
          const storedOriginal = result[FONT_ORIGINAL_SCALE_KEY];
          const storedTranslated = result[FONT_TRANSLATED_SCALE_KEY];
          const legacy = result[LEGACY_FONT_SCALE_KEY];
          const hasOriginal = typeof storedOriginal === "number";
          const hasTranslated = typeof storedTranslated === "number";
          const hasLegacy = typeof legacy === "number";
          if (!hasOriginal && !hasTranslated && hasLegacy) {
            resolve({
              original: legacy,
              translated: legacy,
              persist: true,
            });
            return;
          }
          resolve({
            original: hasOriginal ? storedOriginal : DEFAULT_FONT_SCALE,
            translated: hasTranslated ? storedTranslated : DEFAULT_FONT_SCALE,
            persist: false,
          });
        }
      );
    });
  }

  function saveStoredFontScales(originalScale, translatedScale) {
    if (!chrome?.storage?.local) return;
    chrome.storage.local.set({
      [FONT_ORIGINAL_SCALE_KEY]: originalScale,
      [FONT_TRANSLATED_SCALE_KEY]: translatedScale,
    });
  }

  function loadStoredOverlayOpacity() {
    return new Promise((resolve) => {
      if (!chrome?.storage?.local) {
        resolve(null);
        return;
      }
      chrome.storage.local.get([OVERLAY_OPACITY_KEY], (result) => {
        if (chrome.runtime.lastError) {
          resolve(null);
          return;
        }
        const stored = result[OVERLAY_OPACITY_KEY];
        if (typeof stored === "number") {
          resolve(stored);
          return;
        }
        resolve(null);
      });
    });
  }

  function saveStoredOverlayOpacity(opacity) {
    if (!chrome?.storage?.local) return;
    chrome.storage.local.set({ [OVERLAY_OPACITY_KEY]: opacity });
  }

  function clampFontScale(scale) {
    const numeric = Number(scale);
    if (!Number.isFinite(numeric)) {
      return DEFAULT_FONT_SCALE;
    }
    return Math.min(Math.max(MIN_FONT_SCALE, numeric), MAX_FONT_SCALE);
  }

  function clampOpacity(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return DEFAULT_OVERLAY_OPACITY;
    }
    return Math.min(Math.max(MIN_OVERLAY_OPACITY, numeric), MAX_OVERLAY_OPACITY);
  }

  function applyFontScales(originalScale, translatedScale, persist = false) {
    const clampedOriginal = clampFontScale(originalScale);
    const clampedTranslated = clampFontScale(translatedScale);
    originalFontScale = clampedOriginal;
    translatedFontScale = clampedTranslated;
    if (overlayEl) {
      const originalSize = Math.round(BASE_ORIGINAL_FONT_SIZE * clampedOriginal);
      const translatedSize = Math.round(BASE_TRANSLATED_FONT_SIZE * clampedTranslated);
      overlayEl.style.setProperty("--stt-original-size", `${originalSize}px`);
      overlayEl.style.setProperty("--stt-translated-size", `${translatedSize}px`);
    }
    if (persist) {
      saveStoredFontScales(clampedOriginal, clampedTranslated);
    }
  }

  function applyOverlayOpacity(value, persist = false) {
    const clamped = clampOpacity(value);
    overlayOpacity = clamped;
    if (overlayEl) {
      overlayEl.style.setProperty("--stt-card-alpha", String(clamped));
      const flashAlpha = Math.min(1, clamped + 0.1);
      overlayEl.style.setProperty("--stt-flash-alpha", String(flashAlpha));
    }
    if (persist) {
      saveStoredOverlayOpacity(clamped);
    }
  }

  function clampSize(width, height) {
    const maxWidth = Math.max(MIN_OVERLAY_WIDTH, Math.floor(window.innerWidth * 0.95));
    const maxHeight = Math.max(MIN_OVERLAY_HEIGHT, Math.floor(window.innerHeight * 0.8));
    return {
      width: Math.min(Math.max(MIN_OVERLAY_WIDTH, width), maxWidth),
      height: Math.min(Math.max(MIN_OVERLAY_HEIGHT, height), maxHeight),
    };
  }

  function applyOverlaySize(width, height, persist = false) {
    if (!cardEl) return;
    const clamped = clampSize(width, height);
    overlaySize = { width: clamped.width, height: clamped.height };
    cardEl.style.width = `${clamped.width}px`;
    cardEl.style.height = `${clamped.height}px`;
    if (persist) {
      saveStoredSize(overlaySize);
    }
  }

  function recordOverlaySize(width, height) {
    const clamped = clampSize(width, height);
    overlaySize = { width: clamped.width, height: clamped.height };
  }

  function scheduleSizeSave(width, height) {
    if (resizeSaveTimer) {
      clearTimeout(resizeSaveTimer);
    }
    resizeSaveTimer = setTimeout(() => {
      recordOverlaySize(width, height);
      saveStoredSize(overlaySize);
      resizeSaveTimer = null;
    }, RESIZE_SAVE_DELAY_MS);
  }

  function syncOverlaySizeFromDom() {
    if (!cardEl) return;
    const rect = cardEl.getBoundingClientRect();
    scheduleSizeSave(rect.width, rect.height);
  }

  function clampPosition(x, y) {
    if (!overlayEl) return { x, y };
    const rect = overlayEl.getBoundingClientRect();
    const minX = DRAG_MARGIN;
    const minY = DRAG_MARGIN;
    const maxX = Math.max(minX, window.innerWidth - rect.width - DRAG_MARGIN);
    const maxY = Math.max(minY, window.innerHeight - rect.height - DRAG_MARGIN);
    return {
      x: Math.min(Math.max(minX, x), maxX),
      y: Math.min(Math.max(minY, y), maxY),
    };
  }

  function applyOverlayPosition(x, y, persist = false) {
    if (!overlayEl) return;
    const clamped = clampPosition(x, y);
    overlayPosition = { x: clamped.x, y: clamped.y };
    overlayEl.style.left = `${clamped.x}px`;
    overlayEl.style.top = `${clamped.y}px`;
    overlayEl.style.bottom = "auto";
    overlayEl.style.transform = "none";
    overlayEl.dataset.positioned = "true";
    if (persist) {
      saveStoredPosition(overlayPosition);
    }
  }

  function getEventPoint(event) {
    if (!event) return null;
    if (event.touches && event.touches[0]) {
      return { x: event.touches[0].clientX, y: event.touches[0].clientY };
    }
    if (event.changedTouches && event.changedTouches[0]) {
      return {
        x: event.changedTouches[0].clientX,
        y: event.changedTouches[0].clientY,
      };
    }
    if (typeof event.clientX === "number" && typeof event.clientY === "number") {
      return { x: event.clientX, y: event.clientY };
    }
    return null;
  }

  function isResizeHandle(event) {
    if (!cardEl) return false;
    const point = getEventPoint(event);
    if (!point) return false;
    const rect = cardEl.getBoundingClientRect();
    const nearRight = point.x >= rect.right - RESIZE_HANDLE_SIZE;
    const nearBottom = point.y >= rect.bottom - RESIZE_HANDLE_SIZE;
    return nearRight && nearBottom;
  }

  function ensureAbsolutePosition() {
    if (!overlayEl || overlayEl.dataset.positioned === "true") return;
    const rect = overlayEl.getBoundingClientRect();
    applyOverlayPosition(rect.left, rect.top, false);
  }

  function attachDragHandlers() {
    if (!overlayEl || !cardEl) return;

    if (dragHandlersAttached) return;
    dragHandlersAttached = true;

    document.addEventListener(
      "pointerdown",
      (event) => {
        if (!overlayEl || !cardEl) return;
        if (!overlayEl.classList.contains("visible")) return;
        if (!overlayEl.contains(event.target)) return;
        if (event.pointerType === "mouse" && typeof event.button === "number" && event.button !== 0) {
          return;
        }
        if (isResizeHandle(event)) return;
        const point = getEventPoint(event);
        if (!point) return;
        ensureAbsolutePosition();
        dragState = {
          startX: point.x,
          startY: point.y,
          originX: overlayPosition ? overlayPosition.x : overlayEl.getBoundingClientRect().left,
          originY: overlayPosition ? overlayPosition.y : overlayEl.getBoundingClientRect().top,
          pointerId: typeof event.pointerId === "number" ? event.pointerId : null,
        };
        overlayEl.classList.add("draggable");
        overlayEl.style.setProperty("pointer-events", "auto", "important");
        if (hideTimer) {
          clearTimeout(hideTimer);
          hideTimer = null;
        }
        if (typeof event.pointerId === "number" && cardEl.setPointerCapture) {
          try {
            cardEl.setPointerCapture(event.pointerId);
          } catch (err) {
            // Ignore pointer capture failures.
          }
        }
        event.preventDefault();
      },
      true
    );

    document.addEventListener(
      "pointermove",
      (event) => {
        if (!dragState) return;
        if (
          dragState.pointerId !== null &&
          typeof event.pointerId === "number" &&
          event.pointerId !== dragState.pointerId
        ) {
          return;
        }
        const point = getEventPoint(event);
        if (!point) return;
        const nextX = dragState.originX + (point.x - dragState.startX);
        const nextY = dragState.originY + (point.y - dragState.startY);
        applyOverlayPosition(nextX, nextY, false);
      },
      true
    );

    document.addEventListener(
      "pointerup",
      (event) => {
        if (!dragState) return;
        if (
          dragState.pointerId !== null &&
          typeof event.pointerId === "number" &&
          event.pointerId !== dragState.pointerId
        ) {
          return;
        }
        if (
          cardEl &&
          dragState.pointerId !== null &&
          cardEl.hasPointerCapture(dragState.pointerId)
        ) {
          cardEl.releasePointerCapture(dragState.pointerId);
        }
        overlayEl?.classList.remove("draggable");
        dragState = null;
        if (overlayPosition) {
          saveStoredPosition(overlayPosition);
        }
        if (!resizeObserver) {
          syncOverlaySizeFromDom();
        }
        showOverlay();
      },
      true
    );

    document.addEventListener(
      "pointercancel",
      () => {
        if (!dragState) return;
        overlayEl?.classList.remove("draggable");
        dragState = null;
      },
      true
    );

    window.addEventListener("resize", () => {
      if (overlayPosition) {
        applyOverlayPosition(overlayPosition.x, overlayPosition.y, false);
      }
      if (overlaySize) {
        applyOverlaySize(overlaySize.width, overlaySize.height, true);
      }
    });
  }

  function createOverlay() {
    if (overlayEl) return;

    overlayEl = document.createElement("div");
    overlayEl.id = "stt-subtitle-overlay";

    cardEl = document.createElement("div");
    cardEl.className = "stt-subtitle-card";

    originalEl = document.createElement("div");
    originalEl.className = "stt-original";

    translatedEl = document.createElement("div");
    translatedEl.className = "stt-translated";

    statusEl = document.createElement("div");
    statusEl.className = "stt-status";

    cardEl.appendChild(originalEl);
    cardEl.appendChild(translatedEl);
    cardEl.appendChild(statusEl);
    overlayEl.appendChild(cardEl);

    const host = getOverlayHost();
    if (host) {
      host.appendChild(overlayEl);
    }

    attachDragHandlers();

    if (!positionLoaded) {
      positionLoaded = true;
      loadStoredPosition().then((stored) => {
        if (!stored) return;
        applyOverlayPosition(stored.x, stored.y, false);
      });
    }

    if (!sizeLoaded) {
      sizeLoaded = true;
      applyOverlaySize(DEFAULT_OVERLAY_WIDTH, DEFAULT_OVERLAY_HEIGHT, false);
      loadStoredSize().then((stored) => {
        if (!stored) return;
        applyOverlaySize(stored.width, stored.height, false);
      });
    }

    applyFontScales(originalFontScale, translatedFontScale, false);
    if (!fontScaleLoaded) {
      fontScaleLoaded = true;
      loadStoredFontScales().then((stored) => {
        if (!stored) return;
        applyFontScales(stored.original, stored.translated, stored.persist);
      });
    }

    applyOverlayOpacity(overlayOpacity, false);
    if (!overlayOpacityLoaded) {
      overlayOpacityLoaded = true;
      loadStoredOverlayOpacity().then((stored) => {
        if (stored === null || stored === undefined) return;
        applyOverlayOpacity(stored, false);
      });
    }

    if (!resizeObserver && window.ResizeObserver && cardEl) {
      resizeObserver = new ResizeObserver(() => {
        syncOverlaySizeFromDom();
        if (overlayPosition) {
          applyOverlayPosition(overlayPosition.x, overlayPosition.y, false);
        }
      });
      resizeObserver.observe(cardEl);
    }

    if (!fullscreenListenerAttached) {
      document.addEventListener("fullscreenchange", ensureOverlayHost);
      fullscreenListenerAttached = true;
    }
  }

  function showOverlay() {
    if (!overlayEl) createOverlay();
    overlayEl.classList.add("visible");
    overlayEl.style.setProperty("pointer-events", "auto", "important");

    // 清除之前的隱藏計時器
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }

    // 設定自動隱藏
    if (overlayPosition) {
      applyOverlayPosition(overlayPosition.x, overlayPosition.y, false);
    }
    if (!dragState && HIDE_DELAY_MS > 0) {
      hideTimer = setTimeout(() => {
        hideOverlay();
      }, HIDE_DELAY_MS);
    }
  }

  function flashOverlay() {
    if (!overlayEl) return;
    overlayEl.classList.remove("flash");
    // 強制重排以重啟動畫
    void overlayEl.offsetWidth;
    overlayEl.classList.add("flash");
  }

  function updateSubtitle(original, translated) {
    console.log("[content] updateSubtitle called - overlayEl exists:", !!overlayEl);
    if (!overlayEl) createOverlay();
    ensureOverlayHost();

    const now = Date.now();
    const originalText = original || "";
    const translatedText = translated || "";
    const silenceGap = lastSubtitleTs && now - lastSubtitleTs > SILENCE_CLEAR_MS;
    lastSubtitleTs = now;

    const lengthExceeded =
      originalText.length > MAX_ORIGINAL_CHARS ||
      translatedText.length > MAX_TRANSLATED_CHARS;

    if (silenceGap || lengthExceeded) {
      clearSubtitleText();
    }

    const displayOriginal = trimText(originalText, MAX_ORIGINAL_CHARS);
    const displayTranslated = trimText(translatedText, MAX_TRANSLATED_CHARS);

    if (!displayOriginal && !displayTranslated) {
      return;
    }

    if (displayOriginal) {
      originalEl.textContent = displayOriginal;
      originalEl.style.display = "block";
    } else {
      originalEl.style.display = "none";
    }

    if (displayTranslated) {
      translatedEl.textContent = displayTranslated;
      translatedEl.style.display = "block";
    } else {
      translatedEl.style.display = "none";
    }

    statusEl.style.display = "none";
    showOverlay();
    flashOverlay();
    console.log("[content] updateSubtitle done - overlay visible:", overlayEl?.classList.contains("visible"));
  }

  function updateStatus(message) {
    if (!overlayEl) createOverlay();
    ensureOverlayHost();

    originalEl.style.display = "none";
    translatedEl.style.display = "none";
    statusEl.textContent = message;
    statusEl.style.display = "block";

    showOverlay();
  }

  function hideOverlay() {
    if (overlayEl) {
      overlayEl.classList.remove("visible");
      overlayEl.style.setProperty("pointer-events", "none", "important");
    }
  }

  function removeOverlay() {
    if (overlayEl && overlayEl.parentNode) {
      overlayEl.parentNode.removeChild(overlayEl);
      overlayEl = null;
      cardEl = null;
      originalEl = null;
      translatedEl = null;
      statusEl = null;
    }
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (resizeSaveTimer) {
      clearTimeout(resizeSaveTimer);
      resizeSaveTimer = null;
    }
  }

  // 監聽來自 background.js 的訊息
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("[content] Received message:", message.type, message.original?.slice(0, 30) || message.message || "");

    if (message.type === "subtitle") {
      console.log("[content] Updating subtitle - original:", message.original?.slice(0, 50), "translated:", message.translated?.slice(0, 50));
      updateSubtitle(message.original, message.translated);
      sendResponse({ ok: true });
      return true;
    }

    if (message.type === "subtitle-status") {
      updateStatus(message.message);
      sendResponse({ ok: true });
      return true;
    }

    if (message.type === "subtitle-hide") {
      hideOverlay();
      sendResponse({ ok: true });
      return true;
    }

    if (message.type === "subtitle-remove") {
      removeOverlay();
      sendResponse({ ok: true });
      return true;
    }

    return false;
  });

  if (chrome?.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "local") return;
      const originalChange = changes[FONT_ORIGINAL_SCALE_KEY];
      const translatedChange = changes[FONT_TRANSLATED_SCALE_KEY];
      if (!originalChange && !translatedChange) return;
      const nextOriginal = originalChange ? originalChange.newValue : originalFontScale;
      const nextTranslated = translatedChange ? translatedChange.newValue : translatedFontScale;
      applyFontScales(nextOriginal, nextTranslated, false);
    });
  }

  if (chrome?.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "local") return;
      const opacityChange = changes[OVERLAY_OPACITY_KEY];
      if (!opacityChange) return;
      applyOverlayOpacity(opacityChange.newValue ?? DEFAULT_OVERLAY_OPACITY, false);
    });
  }

  // 初始化時創建 overlay（但不顯示）
  createOverlay();

  console.log("[STT Subtitle] Content script loaded");
})();
