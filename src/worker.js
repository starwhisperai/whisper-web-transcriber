// Whisper Web worker: runs OpenAI Whisper inside the browser with transformers.js.
// The audio arrives as a Float32Array from the page and never leaves this device.
// Model files download once from the Hugging Face hub and stay in the browser cache.
import { pipeline, WhisperTextStreamer, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

env.allowLocalModels = false;

const MODELS = {
  tiny: 'onnx-community/whisper-tiny',
  base: 'onnx-community/whisper-base',
  small: 'onnx-community/whisper-small',
};
const SAMPLE_RATE = 16000;
const CHUNK_S = 30;
const STRIDE_S = 5;

let current = { key: null, asr: null, device: null };
const post = (msg) => self.postMessage(msg);

async function hasWebGPU() {
  try {
    return Boolean(await navigator.gpu?.requestAdapter());
  } catch {
    return false;
  }
}

// WASM: 8-bit weights everywhere (smallest download). WebGPU: fp32 encoder + 4-bit decoder,
// the pair Hugging Face uses in its own WebGPU Whisper demo.
const dtypeFor = (device) => (device === 'webgpu' ? { encoder_model: 'fp32', decoder_model_merged: 'q4' } : 'q8');

async function load(model, want) {
  const device = want !== 'wasm' && (await hasWebGPU()) ? 'webgpu' : 'wasm';
  const key = `${model}|${device}`;
  if (current.key === key) return current;
  if (current.asr) await current.asr.dispose?.();
  current = { key: null, asr: null, device: null };

  const files = new Map();
  const progress_callback = (p) => {
    if (p.status !== 'progress' && p.status !== 'done') return;
    const f = files.get(p.file) || { loaded: 0, total: 0 };
    if (p.total) f.total = p.total;
    f.loaded = p.status === 'done' ? f.total : p.loaded || f.loaded;
    files.set(p.file, f);
    let loaded = 0, total = 0;
    for (const v of files.values()) { loaded += v.loaded; total += v.total; }
    post({ type: 'download', loaded, total });
  };

  const create = (dev) => pipeline('automatic-speech-recognition', MODELS[model], { device: dev, dtype: dtypeFor(dev), progress_callback });
  let asr, used = device;
  try {
    asr = await create(device);
  } catch (err) {
    if (device !== 'webgpu') throw err;
    post({ type: 'notice', message: 'gpu-fallback' });
    used = 'wasm';
    files.clear();
    asr = await create('wasm');
  }
  current = { key: `${model}|${used}`, asr, device: used };
  return current;
}

// Whisper's own language detection: one decoder step after <|startoftranscript|>,
// then the most likely language token. Uses the first 30 seconds.
async function detectLanguage(asr, audio) {
  const { input_features } = await asr.processor(audio.subarray(0, SAMPLE_RATE * CHUNK_S));
  const cfg = asr.model.generation_config;
  const sot = cfg.decoder_start_token_id;
  const decoder_input_ids = new Tensor('int64', BigInt64Array.from([BigInt(sot)]), [1, 1]);
  const { logits } = await asr.model({ input_features, decoder_input_ids });
  const vocab = logits.dims.at(-1);
  const row = logits.data.subarray(logits.data.length - vocab);
  let best = 'en', bestScore = -Infinity;
  for (const [token, id] of Object.entries(cfg.lang_to_id)) {
    if (row[id] > bestScore) { bestScore = row[id]; best = token.slice(2, -2); }
  }
  return best;
}

async function transcribe({ audio, model, language, task, device: want }) {
  post({ type: 'stage', stage: 'loading' });
  const { asr, device } = await load(model, want);
  post({ type: 'ready', device, isolated: self.crossOriginIsolated === true });

  let lang = language;
  if (lang === 'auto') {
    post({ type: 'stage', stage: 'detecting' });
    lang = await detectLanguage(asr, audio);
    post({ type: 'language', language: lang });
  }

  const windowLen = SAMPLE_RATE * CHUNK_S;
  const jump = SAMPLE_RATE * (CHUNK_S - 2 * STRIDE_S);
  const total = audio.length <= windowLen ? 1 : 1 + Math.ceil((audio.length - windowLen) / jump);
  let done = 0;
  post({ type: 'stage', stage: 'transcribing', done, total });
  const streamer = new WhisperTextStreamer(asr.tokenizer, {
    skip_prompt: true,
    callback_function: (text) => post({ type: 'partial', text }),
    on_finalize: () => { done += 1; post({ type: 'stage', stage: 'transcribing', done: Math.min(done, total), total }); },
  });

  const out = await asr(audio, {
    language: lang,
    task,
    chunk_length_s: CHUNK_S,
    stride_length_s: STRIDE_S,
    return_timestamps: true,
    streamer,
  });
  const text = out.text.trim();
  // A GPU driver can return nothing where the CPU path works; try once more on WebAssembly.
  if (!text && device === 'webgpu' && want !== 'webgpu') {
    post({ type: 'notice', message: 'gpu-fallback' });
    return transcribe({ audio, model, language: lang, task, device: 'wasm' });
  }
  post({ type: 'done', text, chunks: out.chunks || [], language: lang, device });
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'transcribe') return;
  try {
    await transcribe(data);
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err) });
  }
};
