// Whisper Web page controller. Reads the file locally, decodes it to 16 kHz mono with Web Audio,
// and hands the samples to worker.js. No file content or transcript is ever sent anywhere.
const $ = (id) => document.getElementById(id);
const els = {
  model: $('wo-model'), language: $('wo-language'), translate: $('wo-translate'),
  file: $('wo-file'), drop: $('wo-drop'), result: $('wo-result'), status: $('wo-status'),
  bar: $('wo-bar'), text: $('wo-text'), stamps: $('wo-stamps'), device: $('wo-device'),
  copy: $('wo-copy'), again: $('wo-again'), cancel: $('wo-cancel'), cta: $('wo-cta'),
};

// Messages, English by default. A localized page sets window.WO_STRINGS before this module runs.
const S = {
  gpu: 'Your browser supports WebGPU, so Whisper will run on your graphics card.',
  cpu: 'No WebGPU here, so Whisper will run on your processor. It works, just slower. Chrome or Edge on a recent computer is fastest.',
  engineFail: 'The transcription engine failed to start.',
  tooBig: 'This file is {mb} MB, more than a browser tab can safely load. Save just the audio as MP3 first, or split the recording.',
  tooBigPhone: 'This file is {mb} MB, more than a browser tab can safely load on a phone. Save just the audio as MP3 first, or split the recording.',
  reading: 'Reading the audio from your file...',
  decodeFail: 'Your browser could not read the audio in this file. Try MP3, WAV, M4A or MP4, or convert it first.',
  codecFail: "This file uses an audio format ({codec}) your browser can't decode. Convert it to MP3 or WAV first.",
  noAudio: 'This file has no audio track we could read.',
  noSpeech: 'No speech found in this file.',
  libFail: "Couldn't download the file reader from cdn.jsdelivr.net. Check your connection. Some company and school networks block it.",
  longCpu: 'This recording is {min} minutes long and your browser is using the processor, so it will take a while. Keep this tab open.',
  loading: 'Loading Whisper...',
  detecting: 'Detecting the language...',
  part: 'Transcribing, part {n} of {total}...',
  download: 'Downloading the model, {loaded} of about {total} MB. This happens once.',
  done: 'Done in {secs} seconds.{lang} You can edit the text below before you copy or download it.',
  lang: ' Language: {name}.',
  downloadFail: "Couldn't download the Whisper model from Hugging Face. Check your connection. Some company and school networks block huggingface.co.",
  memoryFail: 'Your browser ran out of memory. Try the Tiny model, close other tabs, or split the recording into shorter parts.',
  transcribeFail: 'Something went wrong while transcribing ({msg}). Try the Tiny model or a shorter file.',
  copied: 'Copied',
  copy: 'Copy text',
  ...window.WO_STRINGS,
};
const t = (key, vars = {}) => S[key].replace(/\{(\w+)\}/g, (_, k) => vars[k]);

const SIZES = { wasm: { tiny: 41, base: 77, small: 249 }, webgpu: { tiny: 120, base: 206, small: 586 } };
const LANG_NAMES = new Intl.DisplayNames([document.documentElement.lang || 'en'], { type: 'language' });
const FORCED = new URLSearchParams(location.search).get('device'); // 'wasm' or 'webgpu', for testing
const MOBILE = navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const MAX_BYTES = MOBILE ? 200e6 : 1e9; // above this, reading the whole file into memory is not safe
// A 58 MB, 1 hour MP3 peaked at 2.9 GB decoded whole and 525 MB read in pieces, so above this size pieces come first.
const PIECES_ABOVE = 50e6;
const STREAM_LIB = 'https://cdn.jsdelivr.net/npm/mediabunny@1.61.6/+esm'; // needs WebCodecs, loaded only when decodeStreaming runs
const CAN_STREAM = typeof AudioDecoder !== 'undefined';
const PREFER = window.WO_STRINGS ? document.documentElement.lang.slice(0, 2) : null;
let worker = null;
let gpu = false;
let run = null; // { name, seconds, started, chunks, text, language, device }

const track = (name, params) => { try { window.gtag?.('event', name, params); } catch (e) { /* analytics is optional */ } };
const bucket = (min) => (min < 1 ? '<1' : min < 5 ? '1-5' : min < 15 ? '5-15' : min < 60 ? '15-60' : '60+');
const sizeBucket = (bytes) => (bytes < 50e6 ? 'under-50' : bytes < 200e6 ? '50-200' : bytes < 1e9 ? '200-1000' : 'over-1000');
// Optional cookieless counts (<meta name="wo-ping" content="/path/ping.gif">), read from the web server log:
// size bucket and reader at the start, model, device, length bucket and speed at the end, the failed step on an error.
// Never the file or the text.
const PING = document.querySelector('meta[name="wo-ping"]')?.content;
const ping = (e, p) => { if (PING) new Image().src = `${PING}?${new URLSearchParams({ e, ...p, r: Math.random().toString(36).slice(2) })}`; };

const readError = (messageKey, stage = 'decode', codec = '') => Object.assign(new Error(messageKey), { messageKey, stage, codec });

function setState(state, message) {
  els.result.dataset.state = state;
  if (message !== undefined) els.status.textContent = message;
}

function setProgress(fraction) {
  els.bar.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

function updateSizes() {
  const sizes = SIZES[gpu ? 'webgpu' : 'wasm'];
  for (const opt of els.model.options) {
    opt.textContent = `${opt.dataset.label} (${sizes[opt.value]} MB)`;
  }
  els.device.textContent = t(gpu ? 'gpu' : 'cpu');
}

async function detectGPU() {
  try { gpu = FORCED !== 'wasm' && Boolean(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch (e) { gpu = false; }
  if (MOBILE) els.model.querySelector('option[value="small"]')?.remove();
  updateSizes();
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./worker.js?v=7', import.meta.url), { type: 'module' });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => fail('engine', e.message || t('engineFail'));
  }
  return worker;
}

// An OfflineAudioContext at 16 kHz decodes and resamples in one step, needs no audio device
// and keeps memory at 16 kHz instead of the file's own rate.
async function decode(file) {
  try {
    const buf = await file.arrayBuffer();
    let ctx;
    try { ctx = new OfflineAudioContext(1, 1, 16000); } catch (e) { ctx = null; }
    const audio = await (ctx || new OfflineAudioContext(1, 1, 44100)).decodeAudioData(buf);
    const mono = mixDown(audio);
    return audio.sampleRate === 16000 ? mono : resample(mono, audio.sampleRate);
  } catch (e) {
    throw e instanceof RangeError ? readError('memoryFail', 'memory') : e;
  }
}

// Copies, so the samples can be transferred to the worker (AudioBuffer memory is not transferable).
function mixDown(audio) {
  if (audio.numberOfChannels === 1) return new Float32Array(audio.getChannelData(0));
  const n = audio.numberOfChannels;
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < n; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / n;
  }
  return mono;
}

// Fallback for browsers that refuse a 16 kHz context: linear interpolation to 16 kHz.
function resample(input, rate) {
  const ratio = rate / 16000;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const x = i * ratio, j = Math.floor(x), f = x - j;
    out[i] = input[j] * (1 - f) + (input[j + 1] ?? input[j]) * f;
  }
  return out;
}

// Butterworth low-pass in place, three biquads in a row. The rate change below has no filter of its own,
// so without this anything above 8 kHz folds back over the speech.
function lowPass(block, rate) {
  const w = 2 * Math.PI * 7200 / rate, cos = Math.cos(w), sin = Math.sin(w);
  for (const q of [0.5176, 0.7071, 1.9319]) {
    const alpha = sin / (2 * q), a0 = 1 + alpha;
    const b1 = (1 - cos) / a0, b0 = b1 / 2, a1 = -2 * cos / a0, a2 = (1 - alpha) / a0;
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < block.length; i++) {
      const x = block[i], y = b0 * x + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = x; y2 = y1; y1 = y;
      block[i] = y;
    }
  }
}

// One block of mono samples to 16 kHz, with the browser's own resampler.
async function blockTo16k(block, rate) {
  if (rate === 16000) return block;
  if (rate > 16000) lowPass(block, rate);
  const ctx = new OfflineAudioContext(1, Math.max(1, Math.round(block.length * 16000 / rate)), 16000);
  const buffer = ctx.createBuffer(1, block.length, rate);
  buffer.copyToChannel(block, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
  return (await ctx.startRendering()).getChannelData(0);
}

// For files too big to read into memory or that decodeAudioData refuses: reads only the audio track,
// piece by piece, with WebCodecs. The file stays on this device. Returns null if the run was cancelled.
async function decodeStreaming(file, isCurrent) {
  const { Input, BlobSource, ALL_FORMATS, AudioSampleSink } = await import(STREAM_LIB).catch(() => { throw readError('libFail', 'lib'); });
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const track = await input.getPrimaryAudioTrack();
  if (!track) throw readError('noAudio');
  if (!(await track.canDecode())) throw readError('codecFail', 'decode', track.codec || '?');
  // Timestamps can start far from zero, and the duration is where the last one ends.
  const first = await track.getFirstTimestamp();
  const duration = (await track.computeDuration()) - first;
  const samples = (n) => { try { return new Float32Array(n); } catch (e) { throw readError('memoryFail', 'memory'); } };
  let out = samples(Math.ceil(Math.max(0, duration) * 16000) + 16000);
  let written = 0;
  let block = null; // 30 s of mono samples at the decoded rate
  let filled = 0;
  let rate = 0;
  const flush = async () => {
    if (!filled) return;
    const part = await blockTo16k(block.subarray(0, filled), rate);
    if (written + part.length > out.length) { const grown = samples(written + part.length + 16000 * 60); grown.set(out.subarray(0, written)); out = grown; }
    out.set(part, written);
    written += part.length;
    filled = 0;
  };
  for await (const sample of new AudioSampleSink(track).samples()) {
    if (!isCurrent()) { sample.close(); return null; }
    if (sample.sampleRate !== rate) { await flush(); rate = sample.sampleRate; block = new Float32Array(rate * 30); }
    const frames = sample.numberOfFrames, channels = sample.numberOfChannels;
    const mono = new Float32Array(frames);
    const plane = new Float32Array(frames);
    for (let c = 0; c < channels; c++) {
      sample.copyTo(plane, { planeIndex: c, format: 'f32-planar' });
      for (let i = 0; i < frames; i++) mono[i] += plane[i] / channels;
    }
    const seconds = sample.timestamp - first;
    sample.close();
    for (let at = 0; at < frames;) {
      const take = Math.min(frames - at, block.length - filled);
      block.set(mono.subarray(at, at + take), filled);
      filled += take;
      at += take;
      if (filled === block.length) await flush();
    }
    if (duration) setProgress(seconds / duration);
  }
  await flush();
  return out.subarray(0, written);
}

// The first reader that works. Whole decoding after a memory failure in pieces would only fail harder.
async function readAudio(readers, file, isCurrent) {
  const [first, second] = readers;
  try {
    return await first(file, isCurrent);
  } catch (e) {
    if (!second || !isCurrent() || (e.messageKey === 'memoryFail' && second === decode)) throw e;
    try {
      return await second(file, isCurrent);
    } catch (e2) {
      throw e.messageKey ? e : e2;
    }
  }
}

async function start(file) {
  if (!file || els.result.dataset.state === 'working') return;
  const current = { name: file.name.replace(/\.[^.]+$/, '') || 'transcript', started: performance.now(), text: '' };
  const isCurrent = () => run === current;
  run = current;
  els.text.value = '';
  els.text.readOnly = true;
  els.cta.hidden = true;
  setProgress(0);
  setState('working', t('reading'));
  // The piece reader hangs the tab on a damaged FLAC, so FLAC is only ever decoded whole.
  const pieces = CAN_STREAM && (await file.slice(0, 4).text().catch(() => '')) !== 'fLaC';
  if (!isCurrent()) return;
  const big = file.size > MAX_BYTES;
  if (big && !pieces) {
    return fail('size', t(MOBILE ? 'tooBigPhone' : 'tooBig', { mb: Math.round(file.size / 1e6) }));
  }
  const readers = !pieces ? [decode] : big ? [decodeStreaming] : file.size > PIECES_ABOVE ? [decodeStreaming, decode] : [decode, decodeStreaming];
  // Before reading, so a tab that dies on the file still leaves a trace.
  ping('start', { mb: sizeBucket(file.size), via: readers[0] === decode ? 'whole' : 'pieces' });
  let audio;
  try {
    audio = await readAudio(readers, file, isCurrent);
  } catch (e) {
    if (!isCurrent()) return;
    return fail(e.stage || 'decode', t(e.messageKey || 'decodeFail', { codec: e.codec }));
  }
  if (!isCurrent() || !audio) return;
  setProgress(0);
  if (!audio.length) return fail('decode', t('noAudio'));
  run.seconds = audio.length / 16000;
  const model = els.model.value;
  run.model = model;
  track('whisper_web_start', { model, device: gpu ? 'webgpu' : 'wasm', minutes: bucket(run.seconds / 60) });
  getWorker().postMessage({
    type: 'transcribe', audio, model, device: FORCED || (gpu ? 'auto' : 'wasm'),
    language: els.language.value, prefer: PREFER, task: els.translate.checked ? 'translate' : 'transcribe',
  }, [audio.buffer]);
  if (!gpu && run.seconds > 1800) {
    els.device.textContent = t('longCpu', { min: Math.round(run.seconds / 60) });
  }
}

function onWorkerMessage({ data }) {
  if (!run) return;
  switch (data.type) {
    case 'stage':
      if (data.stage === 'loading') setState('working', t('loading'));
      if (data.stage === 'detecting') setState('working', t('detecting'));
      if (data.stage === 'transcribing') {
        setState('working', t('part', { n: Math.min(data.done + 1, data.total), total: data.total }));
        setProgress(data.at);
      }
      break;
    case 'download': {
      // Files report their size only when they start, so the expected size keeps the total steady.
      const total = Math.max(data.total, SIZES[gpu ? 'webgpu' : 'wasm'][run.model] * 1e6);
      setState('working', t('download', { loaded: Math.round(data.loaded / 1e6), total: Math.round(total / 1e6) }));
      setProgress(data.loaded / total);
      break;
    }
    case 'ready':
      window.__woInfo = { device: data.device, isolated: data.isolated, seconds: run.seconds };
      break;
    case 'notice':
      if (data.message === 'gpu-fallback') { gpu = false; updateSizes(); }
      break;
    case 'language':
      run.language = data.language;
      break;
    case 'partial':
      run.text += data.text;
      els.text.value = run.text;
      els.text.scrollTop = els.text.scrollHeight;
      break;
    case 'done': {
      run.chunks = data.chunks;
      run.language = data.language;
      run.device = data.device;
      run.final = data.text;
      render();
      els.text.readOnly = false;
      setProgress(1);
      const secs = (performance.now() - run.started) / 1000;
      const speed = (run.seconds / secs).toFixed(1);
      // Nothing to copy or download, so it shows like a failed run: the message and New file.
      if (!data.text) {
        setState('error', t('noSpeech'));
        ping('empty', { m: run.model, d: data.device, len: bucket(run.seconds / 60), x: speed });
        break;
      }
      const lang = data.language ? t('lang', { name: LANG_NAMES.of(data.language) || data.language }) : '';
      setState('done', t('done', { secs: Math.round(secs), lang }));
      els.cta.hidden = false;
      track('whisper_web_done', { model: run.model, device: data.device, minutes: bucket(run.seconds / 60), speed });
      ping('done', { m: run.model, d: data.device, len: bucket(run.seconds / 60), x: speed });
      break;
    }
    case 'error':
      if (/fetch|network|locate file|load model|404|403/i.test(data.message)) {
        fail('download', t('downloadFail'));
      } else if (/memory|allocation|out of/i.test(data.message)) {
        fail('memory', t('memoryFail'));
      } else {
        fail('transcribe', t('transcribeFail', { msg: data.message }));
      }
      break;
  }
}

function fail(stage, message) {
  track('whisper_web_error', { stage });
  ping('error', { s: stage });
  setProgress(0);
  setState('error', message);
  els.text.readOnly = false;
}

const pad = (n, w = 2) => String(n).padStart(w, '0');
function stamp(sec, sep) {
  const ms = Math.round((sec || 0) * 1000);
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms / 60000) % 60)}:${pad(Math.floor(ms / 1000) % 60)}${sep}${pad(ms % 1000, 3)}`;
}
function segments() {
  return (run?.chunks || [])
    .map((c) => ({ start: c.timestamp[0] ?? 0, end: c.timestamp[1] ?? run.seconds, text: c.text.trim() }))
    .filter((s) => s.text);
}
function render() {
  if (!run || run.final === undefined) return;
  els.text.value = els.stamps.checked
    ? segments().map((s) => `[${stamp(s.start, '.').slice(0, 8)}] ${s.text}`).join('\n')
    : run.final;
}
function exportAs(format) {
  if (format === 'txt') return els.text.value;
  const segs = segments();
  if (format === 'srt') return segs.map((s, i) => `${i + 1}\n${stamp(s.start, ',')} --> ${stamp(s.end, ',')}\n${s.text}\n`).join('\n');
  if (format === 'vtt') return 'WEBVTT\n\n' + segs.map((s) => `${stamp(s.start, '.')} --> ${stamp(s.end, '.')}\n${s.text}\n`).join('\n');
  return '';
}
window.__woExport = exportAs;

function download(format) {
  const blob = new Blob([exportAs(format)], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${run?.name || 'transcript'}.${format}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  track('whisper_web_export', { format });
}

function reset() {
  if (els.result.dataset.state === 'working') {
    worker?.terminate();
    worker = null;
  }
  run = null;
  els.file.value = '';
  els.text.value = '';
  els.cta.hidden = true;
  setProgress(0);
  setState('idle', '');
}

els.file.addEventListener('change', () => start(els.file.files[0]));
els.drop.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('is-over'); });
els.drop.addEventListener('dragleave', () => els.drop.classList.remove('is-over'));
els.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  els.drop.classList.remove('is-over');
  start(e.dataTransfer.files[0]);
});
els.stamps.addEventListener('change', render);
els.copy.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(els.text.value); els.copy.textContent = t('copied'); } catch (e) { els.text.select(); }
  setTimeout(() => { els.copy.textContent = t('copy'); }, 1500);
});
document.querySelectorAll('[data-wo-download]').forEach((b) => b.addEventListener('click', () => download(b.dataset.woDownload)));
els.again.addEventListener('click', reset);
els.cancel.addEventListener('click', reset);
els.cta.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => track('whisper_web_cta', { target: a.dataset.cta })));


detectGPU();
