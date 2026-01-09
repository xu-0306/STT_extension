const DEFAULT_WS_URL = "ws://127.0.0.1:8765/asr";
const FONT_ORIGINAL_SCALE_KEY = "subtitleOriginalScale";
const FONT_TRANSLATED_SCALE_KEY = "subtitleTranslatedScale";
const LEGACY_FONT_SCALE_KEY = "subtitleFontScale";
const TARGET_LANGUAGE_KEY = "translationTargetLanguage";
const STT_MODEL_SIZE_KEY = "sttModelSize";
const OVERLAY_OPACITY_KEY = "subtitleOverlayOpacity";
const DEFAULT_FONT_SCALE = 1;
const MIN_FONT_SCALE = 0.7;
const MAX_FONT_SCALE = 1.6;
const DEFAULT_OVERLAY_OPACITY = 0.85;
const MIN_OVERLAY_OPACITY = 0.35;
const MAX_OVERLAY_OPACITY = 1;
const DEFAULT_STT_MODEL_SIZE = "medium";
const DEFAULT_TARGET_LANGUAGE = "zh-TW";

const DEFAULT_TRANSLATION_MODELS = [
  {
    id: "nllb-600m",
    name: "NLLB 200 (600M)",
    engine: "nllb",
    model: "facebook/nllb-200-distilled-600M",
  },
  {
    id: "ollama-gemma3-1b",
    name: "Ollama gemma3:1b",
    engine: "ollama",
    model: "gemma3:1b",
    host: "http://localhost:11434",
  },
  {
    id: "ollama-gemma3-4b",
    name: "Ollama gemma3:4b",
    engine: "ollama",
    model: "gemma3:4b",
    host: "http://localhost:11434",
  },
  { id: "none", name: "None (no translation)", engine: "noop" },
];

const STT_MODEL_SIZES = [
  { id: "tiny", name: "Tiny" },
  { id: "base", name: "Base" },
  { id: "small", name: "Small" },
  { id: "medium", name: "Medium" },
  { id: "large-v3", name: "Large v3" },
];

const statusEl = document.getElementById("status");
const wsInput = document.getElementById("wsUrl");
const translationModelSelect = document.getElementById("translationModelSelect");
const sttModelSizeSelect = document.getElementById("sttModelSizeSelect");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const openSettingsBtn = document.getElementById("openSettingsBtn");
const subtitleOriginalSizeInput = document.getElementById("subtitleOriginalSize");
const subtitleOriginalSizeValue = document.getElementById("subtitleOriginalSizeValue");
const subtitleTranslatedSizeInput = document.getElementById("subtitleTranslatedSize");
const subtitleTranslatedSizeValue = document.getElementById("subtitleTranslatedSizeValue");
const overlayOpacityInput = document.getElementById("overlayOpacity");
const overlayOpacityValue = document.getElementById("overlayOpacityValue");

const newTranslationNameInput = document.getElementById("newTranslationName");
const newTranslationEngineSelect = document.getElementById("newTranslationEngine");
const newTranslationModelInput = document.getElementById("newTranslationModel");
const newTranslationHostInput = document.getElementById("newTranslationHost");
const newTranslationApiKeyInput = document.getElementById("newTranslationApiKey");
const newTranslationBaseUrlInput = document.getElementById("newTranslationBaseUrl");
const addTranslationModelBtn = document.getElementById("addTranslationModel");
const translationModelList = document.getElementById("translationModelList");
const translationLanguageSelect = document.getElementById("translationLanguageSelect");
const translationLanguageValue = document.getElementById("translationLanguageValue");

const state = {
  translationModels: [],
  selectedTranslationModel: "",
  translationTargetLanguage: DEFAULT_TARGET_LANGUAGE,
  sttModelSize: DEFAULT_STT_MODEL_SIZE,
};

const LANGUAGE_OPTIONS = [
  { id: "auto", name: "Auto" },
  { id: "en", name: "English" },
  { id: "ja", name: "Japanese" },
  { id: "zh-TW", name: "Chinese (Traditional)" },
  { id: "zh-CN", name: "Chinese (Simplified)" },
  { id: "ko", name: "Korean" },
  { id: "fr", name: "French" },
  { id: "de", name: "German" },
  { id: "es", name: "Spanish" },
  { id: "pt", name: "Portuguese" },
  { id: "it", name: "Italian" },
  { id: "ru", name: "Russian" },
  { id: "id", name: "Indonesian" },
  { id: "vi", name: "Vietnamese" },
  { id: "th", name: "Thai" },
];

function setStatus(text, stateName) {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.dataset.state = stateName;
}

function setButtons(isCapturing) {
  if (!startBtn || !stopBtn) return;
  startBtn.disabled = isCapturing;
  stopBtn.disabled = !isCapturing;
}

function storageGet(keys) {
  return new Promise((resolve) => chrome.storage.local.get(keys, resolve));
}

function storageSet(values) {
  return new Promise((resolve) => chrome.storage.local.set(values, resolve));
}

function getActiveTabId() {
  if (!chrome?.tabs?.query) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (chrome.runtime.lastError) {
        resolve(null);
        return;
      }
      resolve(tabs && tabs[0] ? tabs[0].id : null);
    });
  });
}

function setSelectedTranslationId(id) {
  state.selectedTranslationModel = id;
  storageSet({ selectedTranslationModel: id });
}

function buildLabel(model) {
  if (!model.engine) {
    return model.name || "";
  }
  let engineLabel = model.engine;
  if (model.engine === "ollama") {
    engineLabel = "OLLAMA";
  } else if (model.engine === "openai") {
    engineLabel = "OPENAI";
  } else if (model.engine === "nllb") {
    engineLabel = "NLLB";
  } else if (model.engine === "noop") {
    engineLabel = "NONE";
  }
  return `${model.name} [${engineLabel}]`;
}

function renderSelect(selectEl, models, selectedId) {
  if (!selectEl) return;
  selectEl.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = buildLabel(model);
    selectEl.appendChild(option);
  });
  if (selectedId) {
    selectEl.value = selectedId;
  }
}

function renderTranslationSelect() {
  const selectedId = state.selectedTranslationModel || state.translationModels[0]?.id || "";
  renderSelect(translationModelSelect, state.translationModels, selectedId);
}

function renderSttModelSizeSelect() {
  renderSelect(sttModelSizeSelect, STT_MODEL_SIZES, state.sttModelSize);
}

function renderTranslationLanguageSelect() {
  if (!translationLanguageSelect) return;
  renderSelect(translationLanguageSelect, LANGUAGE_OPTIONS, state.translationTargetLanguage);
  const selected =
    LANGUAGE_OPTIONS.find((option) => option.id === state.translationTargetLanguage) ||
    LANGUAGE_OPTIONS[0];
  if (translationLanguageValue) {
    translationLanguageValue.textContent = selected?.name || "Auto";
  }
}

function clampFontScale(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_FONT_SCALE;
  return Math.min(Math.max(MIN_FONT_SCALE, numeric), MAX_FONT_SCALE);
}

function formatFontScale(scale) {
  return `${Math.round(scale * 100)}%`;
}

function updateScaleUI(inputEl, valueEl, scale) {
  const clamped = clampFontScale(scale);
  if (!inputEl) return clamped;
  inputEl.value = String(clamped);
  if (valueEl) {
    valueEl.textContent = formatFontScale(clamped);
  }
  return clamped;
}

function clampOpacity(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_OVERLAY_OPACITY;
  return Math.min(Math.max(MIN_OVERLAY_OPACITY, numeric), MAX_OVERLAY_OPACITY);
}

function updateOpacityUI(inputEl, valueEl, opacity) {
  const clamped = clampOpacity(opacity);
  if (!inputEl) return clamped;
  inputEl.value = String(clamped);
  if (valueEl) {
    valueEl.textContent = `${Math.round(clamped * 100)}%`;
  }
  return clamped;
}

function renderModelList(container, models, onUse, onRemove, onTest) {
  if (!container) return;
  container.innerHTML = "";
  models.forEach((model) => {
    const item = document.createElement("div");
    item.className = "model-item";

    const meta = document.createElement("div");
    meta.className = "model-meta";

    const name = document.createElement("div");
    name.className = "model-name";
    name.textContent = model.name;

    const detail = document.createElement("div");
    detail.className = "model-detail";
    if (model.engine === "ollama") {
      detail.textContent = `${model.engine} · ${model.model || "model"} · ${model.host || ""}`;
    } else if (model.engine === "openai") {
      detail.textContent = `${model.engine} · ${model.model || "model"} · api key ${model.apiKey ? "set" : "empty"}`;
    } else if (model.engine === "nllb") {
      detail.textContent = `${model.engine} · ${model.model || "default"}`;
    } else {
      detail.textContent = `${model.engine}`;
    }

    meta.appendChild(name);
    meta.appendChild(detail);

    const actions = document.createElement("div");
    actions.style.display = "flex";
    actions.style.gap = "6px";

    if (typeof onTest === "function") {
      const testBtn = document.createElement("button");
      testBtn.className = "mini";
      testBtn.textContent = "Test";
      testBtn.addEventListener("click", () => onTest(model));
      actions.appendChild(testBtn);
    }
    const useBtn = document.createElement("button");
    useBtn.className = "mini";
    useBtn.textContent = "Use";
    useBtn.addEventListener("click", () => onUse(model));

    const removeBtn = document.createElement("button");
    removeBtn.className = "mini danger";
    removeBtn.textContent = "Delete";
    removeBtn.addEventListener("click", () => onRemove(model));

    actions.appendChild(useBtn);
    actions.appendChild(removeBtn);

    item.appendChild(meta);
    item.appendChild(actions);
    container.appendChild(item);
  });
}

function ensureDefaults(data) {
  let changed = false;
  if (!Array.isArray(data.translationModels) || data.translationModels.length === 0) {
    data.translationModels = DEFAULT_TRANSLATION_MODELS.slice();
    changed = true;
  } else {
    const beforeCount = data.translationModels.length;
    data.translationModels = data.translationModels.filter(
      (model) => model?.id !== "openai-gpt-4o-mini"
    );
    if (data.translationModels.length !== beforeCount) {
      changed = true;
    }
    const existingIds = new Set(data.translationModels.map((model) => model.id));
    const gemma1b = DEFAULT_TRANSLATION_MODELS.find(
      (model) => model.id === "ollama-gemma3-1b"
    );
    if (gemma1b && !existingIds.has(gemma1b.id)) {
      data.translationModels = [...data.translationModels, gemma1b];
      changed = true;
    }
  }
  return { data, changed };
}

async function loadState() {
  const data = await storageGet([
    "wsUrl",
    "translationModels",
    "selectedTranslationModel",
    FONT_ORIGINAL_SCALE_KEY,
    FONT_TRANSLATED_SCALE_KEY,
    LEGACY_FONT_SCALE_KEY,
    TARGET_LANGUAGE_KEY,
    STT_MODEL_SIZE_KEY,
    OVERLAY_OPACITY_KEY,
  ]);

  const { data: normalized, changed } = ensureDefaults(data);
  if (changed) {
    await storageSet({
      translationModels: normalized.translationModels,
    });
  }

  wsInput.value = normalized.wsUrl || DEFAULT_WS_URL;
  state.translationModels = normalized.translationModels || DEFAULT_TRANSLATION_MODELS.slice();
  state.selectedTranslationModel = normalized.selectedTranslationModel || "";
  state.translationTargetLanguage = data[TARGET_LANGUAGE_KEY] || DEFAULT_TARGET_LANGUAGE;
  state.sttModelSize = data[STT_MODEL_SIZE_KEY] || DEFAULT_STT_MODEL_SIZE;

  renderTranslationSelect();
  renderSttModelSizeSelect();
  renderTranslationLanguageSelect();
  const legacyScale = data[LEGACY_FONT_SCALE_KEY];
  const hasLegacy = typeof legacyScale === "number";
  const originalScale =
    data[FONT_ORIGINAL_SCALE_KEY] ??
    (hasLegacy ? legacyScale : DEFAULT_FONT_SCALE);
  const translatedScale =
    data[FONT_TRANSLATED_SCALE_KEY] ??
    (hasLegacy ? legacyScale : DEFAULT_FONT_SCALE);
  const clampedOriginal = updateScaleUI(
    subtitleOriginalSizeInput,
    subtitleOriginalSizeValue,
    originalScale
  );
  const clampedTranslated = updateScaleUI(
    subtitleTranslatedSizeInput,
    subtitleTranslatedSizeValue,
    translatedScale
  );
  const opacityValue = updateOpacityUI(
    overlayOpacityInput,
    overlayOpacityValue,
    data[OVERLAY_OPACITY_KEY] ?? DEFAULT_OVERLAY_OPACITY
  );
  if (hasLegacy && (data[FONT_ORIGINAL_SCALE_KEY] === undefined || data[FONT_TRANSLATED_SCALE_KEY] === undefined)) {
    storageSet({
      [FONT_ORIGINAL_SCALE_KEY]: clampedOriginal,
      [FONT_TRANSLATED_SCALE_KEY]: clampedTranslated,
    });
  }
  if (data[OVERLAY_OPACITY_KEY] === undefined) {
    storageSet({ [OVERLAY_OPACITY_KEY]: opacityValue });
  }
  if (data[STT_MODEL_SIZE_KEY] === undefined) {
    storageSet({ [STT_MODEL_SIZE_KEY]: state.sttModelSize });
  }
  if (data[TARGET_LANGUAGE_KEY] === undefined) {
    storageSet({ [TARGET_LANGUAGE_KEY]: state.translationTargetLanguage });
  }
  renderModelList(
    translationModelList,
    state.translationModels,
    (model) => {
      setSelectedTranslationId(model.id);
      renderTranslationSelect();
    },
    (model) => removeTranslationModel(model.id),
    (model) => testTranslationModel(model)
  );
}

async function refreshCaptureStatus() {
  const tabId = await getActiveTabId();
  chrome.runtime.sendMessage({ type: "popup-status", tabId }, (response) => {
    if (chrome.runtime.lastError) {
      return;
    }
    if (!response || !response.ok) {
      return;
    }
    const capturing = Boolean(response.capturing);
    const globalCapturing = Boolean(response.globalCapturing);
    let statusText = response.status || (capturing ? "Capturing" : "Idle");
    if (!capturing && globalCapturing) {
      statusText = "Capturing in another tab";
    }
    setButtons(capturing);
    setStatus(statusText, capturing ? "active" : "idle");
  });
}

function runWsTest(payload, label) {
  const wsUrl = wsInput.value.trim() || DEFAULT_WS_URL;
  setStatus(`Testing ${label}...`, "active");

  return new Promise((resolve) => {
    let finished = false;
    const ws = new WebSocket(wsUrl);
    const timeout = window.setTimeout(() => {
      if (finished) return;
      finished = true;
      ws.close();
      resolve({ ok: false, message: "timeout" });
    }, 15000);

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify(payload));
    });

    ws.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (data.type === "test_result") {
        if (finished) return;
        finished = true;
        window.clearTimeout(timeout);
        ws.close();
        resolve(data);
      }
    });

    ws.addEventListener("error", () => {
      if (finished) return;
      finished = true;
      window.clearTimeout(timeout);
      resolve({ ok: false, message: "connection error" });
    });
  }).then((result) => {
    if (result.ok) {
      setStatus(result.message || "Test ok", "active");
    } else {
      setStatus(`Test failed: ${result.message || "error"}`, "error");
    }
  });
}

function testTranslationModel(model) {
  const translation = { engine: model.engine };
  if (state.translationTargetLanguage && state.translationTargetLanguage !== "auto") {
    translation.target_language = state.translationTargetLanguage;
  }
  if (model.engine === "nllb") {
    translation.nllb = { model: model.model };
  }
  if (model.engine === "ollama") {
    translation.ollama = {
      model: model.model,
      host: model.host,
    };
  }
  if (model.engine === "openai") {
    translation.openai = {
      api_key: model.apiKey,
      model: model.model,
      base_url: model.baseUrl,
    };
  }
  const payload = {
    type: "test",
    target: "translation",
    translation,
  };
  return runWsTest(payload, "Translation");
}

function removeTranslationModel(modelId) {
  state.translationModels = state.translationModels.filter((model) => model.id !== modelId);
  if (state.translationModels.length === 0) {
    state.translationModels = DEFAULT_TRANSLATION_MODELS.slice();
  }
  storageSet({ translationModels: state.translationModels });
  if (!state.translationModels.find((model) => model.id === state.selectedTranslationModel)) {
    setSelectedTranslationId(state.translationModels[0].id);
  }
  renderTranslationSelect();
  renderModelList(
    translationModelList,
    state.translationModels,
    (model) => {
      setSelectedTranslationId(model.id);
      renderTranslationSelect();
    },
    (model) => removeTranslationModel(model.id),
    (model) => testTranslationModel(model)
  );
}

function makeId(prefix, name) {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
  return `${prefix}-${slug}-${Date.now()}`;
}

translationModelSelect.addEventListener("change", () => {
  setSelectedTranslationId(translationModelSelect.value);
});

if (sttModelSizeSelect) {
  sttModelSizeSelect.addEventListener("change", () => {
    const value = sttModelSizeSelect.value || DEFAULT_STT_MODEL_SIZE;
    state.sttModelSize = value;
    storageSet({ [STT_MODEL_SIZE_KEY]: value });
  });
}

if (translationLanguageSelect) {
  translationLanguageSelect.addEventListener("change", () => {
    const value = translationLanguageSelect.value || "auto";
    state.translationTargetLanguage = value;
    storageSet({ [TARGET_LANGUAGE_KEY]: value });
    renderTranslationLanguageSelect();
  });
}

if (subtitleOriginalSizeInput) {
  subtitleOriginalSizeInput.addEventListener("input", () => {
    const scale = clampFontScale(subtitleOriginalSizeInput.value);
    updateScaleUI(subtitleOriginalSizeInput, subtitleOriginalSizeValue, scale);
    storageSet({ [FONT_ORIGINAL_SCALE_KEY]: scale });
  });
}

if (subtitleTranslatedSizeInput) {
  subtitleTranslatedSizeInput.addEventListener("input", () => {
    const scale = clampFontScale(subtitleTranslatedSizeInput.value);
    updateScaleUI(subtitleTranslatedSizeInput, subtitleTranslatedSizeValue, scale);
    storageSet({ [FONT_TRANSLATED_SCALE_KEY]: scale });
  });
}

if (overlayOpacityInput) {
  overlayOpacityInput.addEventListener("input", () => {
    const opacity = clampOpacity(overlayOpacityInput.value);
    updateOpacityUI(overlayOpacityInput, overlayOpacityValue, opacity);
    storageSet({ [OVERLAY_OPACITY_KEY]: opacity });
  });
}

function normalizeTranslationForm() {
  if (
    !newTranslationEngineSelect ||
    !newTranslationModelInput ||
    !newTranslationHostInput ||
    !newTranslationApiKeyInput ||
    !newTranslationBaseUrlInput
  ) {
    return;
  }
  const engine = newTranslationEngineSelect.value;
  const usesModel = engine !== "noop";
  newTranslationModelInput.disabled = !usesModel;
  newTranslationHostInput.disabled = engine !== "ollama";
  newTranslationApiKeyInput.disabled = engine !== "openai";
  newTranslationBaseUrlInput.disabled = engine !== "openai";
}

if (newTranslationEngineSelect) {
  newTranslationEngineSelect.addEventListener("change", normalizeTranslationForm);
}

if (addTranslationModelBtn) {
  addTranslationModelBtn.addEventListener("click", () => {
    if (
      !newTranslationNameInput ||
      !newTranslationEngineSelect ||
      !newTranslationModelInput ||
      !newTranslationHostInput ||
      !newTranslationApiKeyInput ||
      !newTranslationBaseUrlInput
    ) {
      return;
    }
  const name = newTranslationNameInput.value.trim();
  const engine = newTranslationEngineSelect.value.trim();
  if (!name) {
    setStatus("Translation name required", "error");
    return;
  }
  const model = newTranslationModelInput.value.trim();
  const host = newTranslationHostInput.value.trim();
  const apiKey = newTranslationApiKeyInput.value.trim();
  const baseUrl = newTranslationBaseUrlInput.value.trim();
  const newModel = {
    id: makeId("translation", name),
    name,
    engine,
    model,
    host,
    apiKey,
    baseUrl,
  };
  state.translationModels = [...state.translationModels, newModel];
  storageSet({ translationModels: state.translationModels });
  newTranslationNameInput.value = "";
  newTranslationModelInput.value = "";
  newTranslationHostInput.value = "";
  newTranslationApiKeyInput.value = "";
  newTranslationBaseUrlInput.value = "";
  setStatus("Translation model added", "active");
  renderTranslationSelect();
  renderModelList(
    translationModelList,
    state.translationModels,
    (modelItem) => {
      setSelectedTranslationId(modelItem.id);
      renderTranslationSelect();
    },
    (modelItem) => removeTranslationModel(modelItem.id),
    (modelItem) => testTranslationModel(modelItem)
  );
  });
}

if (openSettingsBtn) {
  openSettingsBtn.addEventListener("click", () => {
    if (chrome.runtime.openOptionsPage) {
      chrome.runtime.openOptionsPage();
      return;
    }
    setStatus("Options page not available", "error");
  });
}

if (startBtn) {
  startBtn.addEventListener("click", async () => {
    const tabId = await getActiveTabId();
    if (!tabId) {
      setStatus("Error: no active tab", "error");
      return;
    }
    const wsUrl = wsInput.value.trim() || DEFAULT_WS_URL;
    const translationModel =
      state.translationModels.find((model) => model.id === state.selectedTranslationModel) ||
      state.translationModels[0];

    const translation = { engine: translationModel?.engine || "nllb" };
    if (state.translationTargetLanguage && state.translationTargetLanguage !== "auto") {
      translation.target_language = state.translationTargetLanguage;
    }
    if (translationModel?.engine === "nllb") {
      translation.nllb = { model: translationModel.model };
    }
    if (translationModel?.engine === "ollama") {
      translation.ollama = {
        model: translationModel.model,
        host: translationModel.host || "http://localhost:11434",
      };
    }
    if (translationModel?.engine === "openai") {
      translation.openai = {
        api_key: translationModel.apiKey || "",
        model: translationModel.model || "gpt-4o-mini",
        base_url: translationModel.baseUrl || "",
      };
    }
    const settings = { translation, stt: { model: state.sttModelSize } };

    setStatus("Starting...", "active");
    setButtons(true);

    storageSet({ wsUrl });

    chrome.runtime.sendMessage(
      { type: "popup-start", tabId, wsUrl, settings },
      (response) => {
        if (chrome.runtime.lastError) {
          setStatus("Error: extension unavailable", "error");
          setButtons(false);
          return;
        }
        if (!response || !response.ok) {
          setStatus(`Error: ${response?.error || "failed"}`, "error");
          setButtons(false);
          return;
        }
        setStatus("Capturing", "active");
      }
    );
  });
}

if (stopBtn) {
  stopBtn.addEventListener("click", async () => {
    const tabId = await getActiveTabId();
    setStatus("Stopping...", "active");

    chrome.runtime.sendMessage({ type: "popup-stop", tabId }, (response) => {
      if (chrome.runtime.lastError) {
        setStatus("Error: extension unavailable", "error");
        return;
      }
      if (!response || !response.ok) {
        setStatus(`Error: ${response?.error || "failed"}`, "error");
        return;
      }
      setStatus("Idle", "idle");
      setButtons(false);
    });
  });
}

normalizeTranslationForm();
loadState().then(refreshCaptureStatus);
