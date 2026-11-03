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
const MIN_PIECE_S = 20;
const MAX_PIECE_S = 29;
const PAUSE_S = 0.3;
const STOPPED_EARLY_S = 1.5;
const PREFER_MIN = 0.05; // English speech scored the page language at 2.2% at most, noise and tones up to 15%.
const SILENT_PEAK = 1e-4; // 3 steps of 16-bit audio. Whisper writes "you" on digital silence, 150 times in 5 minutes.

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
async function detectLanguage(asr, audio, prefer) {
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
  // A localized page's own language wins a close call, such as Portuguese against Spanish.
  const preferred = cfg.lang_to_id[`<|${prefer}|>`];
  if (preferred !== undefined) {
    let total = 0;
    for (const id of Object.values(cfg.lang_to_id)) total += Math.exp(row[id] - bestScore);
    if (Math.exp(row[preferred] - bestScore) / total >= PREFER_MIN) return prefer;
  }
  return best;
}

// Whisper hears 30 s at a time. Overlapping 30 s windows repeat or lose sentences where they meet, so a piece ends
// at the quietest moment between 20 and 29 s from its start, and the next piece starts there, with no overlap.
function nextCut(audio, start) {
  const from = start + SAMPLE_RATE * MIN_PIECE_S, to = start + SAMPLE_RATE * MAX_PIECE_S, win = Math.round(SAMPLE_RATE * PAUSE_S);
  if (audio.length <= to) return audio.length;
  let cut = to, quietest = Infinity, energy = 0;
  for (let i = from; i < to; i++) {
    energy += audio[i] * audio[i];
    if (i - from >= win) energy -= audio[i - win] * audio[i - win];
    if (i - from >= win - 1 && energy < quietest) { quietest = energy; cut = i - (win >> 1); }
  }
  return cut;
}

function peakOf(piece) {
  let peak = 0;
  for (let i = 0; i < piece.length; i += 16) peak = Math.max(peak, Math.abs(piece[i]));
  return peak;
}

// A piece where the same few characters keep coming back: Whisper is stuck, as in "4. 4. 4.", and the speech under it is lost.
function looped(text) {
  const s = text.replace(/\s+/g, '');
  if (s.length < 40) return false;
  const seen = new Set();
  for (let i = 0; i + 4 <= s.length; i++) seen.add(s.slice(i, i + 4));
  return seen.size / (s.length - 3) < 0.3;
}

async function transcribe({ audio, model, language, prefer, task, device: want }) {
  post({ type: 'stage', stage: 'loading' });
  const { asr, device } = await load(model, want);
  post({ type: 'ready', device, isolated: self.crossOriginIsolated === true });

  let lang = language;
  if (lang === 'auto') {
    post({ type: 'stage', stage: 'detecting' });
    lang = await detectLanguage(asr, audio, prefer);
    post({ type: 'language', language: lang });
  }

  const piecesLeft = (from) => Math.ceil((audio.length - from) / (SAMPLE_RATE * MAX_PIECE_S));
  const decodePiece = (piece, extra) => asr(piece, {
    language: lang, task, return_timestamps: true, ...extra,
    streamer: new WhisperTextStreamer(asr.tokenizer, {
      skip_prompt: true,
      callback_function: (text) => post({ type: 'partial', text }),
    }),
  });
  let raw = '';
  const chunks = [];
  let from = 0, done = 0, decoded = 0, resumed = false;
  const progress = () => post({ type: 'stage', stage: 'transcribing', done, total: done + piecesLeft(from), at: from / audio.length });
  progress();
  while (from < audio.length) {
    const to = nextCut(audio, from);
    const piece = audio.subarray(from, to);
    if (peakOf(piece) < SILENT_PEAK) {
      from = to;
      resumed = false;
      done += 1;
      progress();
      continue;
    }
    const offset = from / SAMPLE_RATE, length = (to - from) / SAMPLE_RATE;
    let out = await decodePiece(piece);
    if (looped(out.text)) out = await decodePiece(piece, { no_repeat_ngram_size: 3 });
    decoded += 1;
    raw += out.text;
    let heard = 0;
    for (const c of out.chunks || []) {
      // Whisper counts up to 30 s even in a shorter piece, so a cue can end past the piece and overlap the next one.
      heard = Math.min(c.timestamp[1] ?? length, length);
      chunks.push({ text: c.text, timestamp: [offset + Math.min(c.timestamp[0] ?? 0, heard), offset + heard] });
    }
    // Whisper can stop decoding before the piece ends. Carry on from the last moment it reached, or the rest is lost.
    // Only once in a row: on music it stops early every time, and 5 minutes became 150 pieces.
    const reached = from + Math.round(heard * SAMPLE_RATE);
    const resume = !resumed && heard > STOPPED_EARLY_S && to - reached > SAMPLE_RATE * STOPPED_EARLY_S;
    from = resume ? reached : to;
    resumed = resume;
    done += 1;
    progress();
  }
  const text = raw.trim();
  // A GPU driver can return nothing where the CPU path works; try once more on WebAssembly.
  if (!text && decoded && device === 'webgpu' && want !== 'webgpu') {
    post({ type: 'notice', message: 'gpu-fallback' });
    return transcribe({ audio, model, language: lang, task, device: 'wasm' });
  }
  post({ type: 'done', text, chunks, language: lang, device });
}

self.onmessage = async ({ data }) => {
  if (data.type !== 'transcribe') return;
  try {
    await transcribe(data);
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err) });
  }
};
