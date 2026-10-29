// Whisper Web page controller. Reads the file locally, decodes it to 16 kHz mono with Web Audio,
// and hands the samples to worker.js. No file content or transcript is ever sent anywhere.
const $ = (id) => document.getElementById(id);
const els = {
  model: $('wo-model'), language: $('wo-language'), translate: $('wo-translate'),
  file: $('wo-file'), drop: $('wo-drop'), result: $('wo-result'), status: $('wo-status'),
  bar: $('wo-bar'), text: $('wo-text'), stamps: $('wo-stamps'), device: $('wo-device'),
  copy: $('wo-copy'), again: $('wo-again'), cancel: $('wo-cancel'), cta: $('wo-cta'),
};

const SIZES = { wasm: { tiny: 41, base: 77, small: 249 }, webgpu: { tiny: 120, base: 206, small: 586 } };
const LANG_NAMES = new Intl.DisplayNames(['en'], { type: 'language' });
const FORCED = new URLSearchParams(location.search).get('device'); // 'wasm' or 'webgpu', for testing
const MOBILE = navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const MAX_BYTES = MOBILE ? 200e6 : 1e9; // the whole file is read into memory before decoding
let worker = null;
let gpu = false;
let run = null; // { name, seconds, started, chunks, text, language, device }

const track = (name, params) => { try { window.gtag?.('event', name, params); } catch (e) { /* analytics is optional */ } };
const bucket = (min) => (min < 1 ? '<1' : min < 5 ? '1-5' : min < 15 ? '5-15' : min < 60 ? '15-60' : '60+');
// Optional cookieless count of finished runs (<meta name="wo-ping" content="/path/ping.gif">), read from the
// web server log: model, device, length bucket. Never the file or the text.
const PING = document.querySelector('meta[name="wo-ping"]')?.content;
const ping = (e, p) => { if (PING) new Image().src = `${PING}?${new URLSearchParams({ e, ...p, r: Math.random().toString(36).slice(2) })}`; };

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
  els.device.textContent = gpu
    ? 'Your browser supports WebGPU, so Whisper will run on your graphics card.'
    : 'No WebGPU here, so Whisper will run on your processor. It works, just slower. Chrome or Edge on a recent computer is fastest.';
}

async function detectGPU() {
  try { gpu = FORCED !== 'wasm' && Boolean(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch (e) { gpu = false; }
  if (MOBILE) els.model.querySelector('option[value="small"]')?.remove();
  updateSizes();
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./worker.js?v=4', import.meta.url), { type: 'module' });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => fail('engine', e.message || 'The transcription engine failed to start.');
  }
  return worker;
}

// An OfflineAudioContext at 16 kHz decodes and resamples in one step, needs no audio device
// and keeps memory at 16 kHz instead of the file's own rate.
async function decode(file) {
  const buf = await file.arrayBuffer();
  let ctx;
  try { ctx = new OfflineAudioContext(1, 1, 16000); } catch (e) { ctx = null; }
  const audio = await (ctx || new OfflineAudioContext(1, 1, 44100)).decodeAudioData(buf);
  const mono = mixDown(audio);
  return audio.sampleRate === 16000 ? mono : resample(mono, audio.sampleRate);
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

async function start(file) {
  if (!file || els.result.dataset.state === 'working') return;
  run = { name: file.name.replace(/\.[^.]+$/, '') || 'transcript', started: performance.now(), text: '' };
  els.text.value = '';
  els.text.readOnly = true;
  els.cta.hidden = true;
  setProgress(0);
  if (file.size > MAX_BYTES) {
    return fail('size', `This file is ${Math.round(file.size / 1e6)} MB, more than a browser tab can safely load${MOBILE ? ' on a phone' : ''}. Save just the audio as MP3 first, or split the recording.`);
  }
  setState('working', 'Reading the audio from your file...');
  let audio;
  try {
    audio = await decode(file);
  } catch (e) {
    return fail('decode', 'Your browser could not read the audio in this file. Try MP3, WAV, M4A or MP4, or convert it first.');
  }
  if (!audio.length) return fail('decode', 'This file has no audio track we could read.');
  run.seconds = audio.length / 16000;
  const model = els.model.value;
  run.model = model;
  track('whisper_web_start', { model, device: gpu ? 'webgpu' : 'wasm', minutes: bucket(run.seconds / 60) });
  getWorker().postMessage({
    type: 'transcribe', audio, model, device: FORCED || (gpu ? 'auto' : 'wasm'),
    language: els.language.value, task: els.translate.checked ? 'translate' : 'transcribe',
  }, [audio.buffer]);
  if (!gpu && run.seconds > 1800) {
    els.device.textContent = `This recording is ${Math.round(run.seconds / 60)} minutes long and your browser is using the processor, so it will take a while. Keep this tab open.`;
  }
}

function onWorkerMessage({ data }) {
  if (!run) return;
  switch (data.type) {
    case 'stage':
      if (data.stage === 'loading') setState('working', 'Loading Whisper...');
      if (data.stage === 'detecting') setState('working', 'Detecting the language...');
      if (data.stage === 'transcribing') {
        setState('working', `Transcribing, part ${Math.min(data.done + 1, data.total)} of ${data.total}...`);
        setProgress(data.done / data.total);
      }
      break;
    case 'download': {
      // Files report their size only when they start, so the expected size keeps the total steady.
      const total = Math.max(data.total, SIZES[gpu ? 'webgpu' : 'wasm'][run.model] * 1e6);
      setState('working', `Downloading the model, ${Math.round(data.loaded / 1e6)} of about ${Math.round(total / 1e6)} MB. This happens once.`);
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
      const lang = data.language ? ` Language: ${LANG_NAMES.of(data.language) || data.language}.` : '';
      setState('done', `Done in ${Math.round(secs)} seconds.${lang} You can edit the text below before you copy or download it.`);
      els.cta.hidden = false;
      const speed = (run.seconds / secs).toFixed(1);
      track('whisper_web_done', { model: run.model, device: data.device, minutes: bucket(run.seconds / 60), speed });
      ping('done', { m: run.model, d: data.device, len: bucket(run.seconds / 60), x: speed });
      break;
    }
    case 'error':
      if (/fetch|network|locate file|load model|404|403/i.test(data.message)) {
        fail('download', "Couldn't download the Whisper model from Hugging Face. Check your connection. Some company and school networks block huggingface.co.");
      } else if (/memory|allocation|out of/i.test(data.message)) {
        fail('memory', 'Your browser ran out of memory. Try the Tiny model, close other tabs, or split the recording into shorter parts.');
      } else {
        fail('transcribe', `Something went wrong while transcribing (${data.message}). Try the Tiny model or a shorter file.`);
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
  try { await navigator.clipboard.writeText(els.text.value); els.copy.textContent = 'Copied'; } catch (e) { els.text.select(); }
  setTimeout(() => { els.copy.textContent = 'Copy text'; }, 1500);
});
document.querySelectorAll('[data-wo-download]').forEach((b) => b.addEventListener('click', () => download(b.dataset.woDownload)));
els.again.addEventListener('click', reset);
els.cancel.addEventListener('click', () => {
  worker?.terminate();
  worker = null;
  reset();
});
els.cta.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => track('whisper_web_cta', { target: a.dataset.cta })));

detectGPU();
