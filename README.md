# Whisper Web Transcriber

Transcribe audio and video files with OpenAI's Whisper inside the browser. The file is decoded and transcribed on the visitor's own machine. Nothing is uploaded.

Live version: https://starwhisper.ai/whisper-web.html

- Runs Whisper tiny, base or small through [transformers.js](https://github.com/huggingface/transformers.js)
- Uses WebGPU when the browser has it, WebAssembly otherwise
- Detects the spoken language, or you pick it; can translate into English
- Exports plain text, SRT and VTT
- Reads MP3, WAV, M4A, OGG, Opus, FLAC and WEBM audio, plus the audio track of MP4, MOV, MKV and WEBM video
- No build step, no server code, no dependencies to install

## How it works

1. `src/app.js` decodes the file to 16 kHz mono. Files up to 50 MB go through `decodeAudioData`. Larger files are read in pieces with [Mediabunny](https://github.com/Vanilagy/mediabunny) and WebCodecs, so memory follows the length of the audio and not the size of the file, and `decodeAudioData` is the fallback up to 1 GB on a computer and 200 MB on a phone. FLAC is only ever decoded whole, because a damaged FLAC can hang the piece reader.
2. The samples are transferred to `src/worker.js`, a module worker that loads transformers.js from jsDelivr and the ONNX model from the Hugging Face hub. The browser caches both after the first run.
3. Long recordings are cut at the quietest moment between 20 and 29 seconds, with no overlap. If Whisper stops before the end of a piece, the next piece starts from the last moment it reached, once in a row at most. Digital silence is skipped, and a piece where Whisper gets stuck repeating itself is decoded again with `no_repeat_ngram_size`. Overlapping 30 second windows wrote sentences twice where they met: on an 8 minute LibriVox clip with Base on WebGPU, cutting at pauses took word error from 8.08% to 5.15% and repeated 6 word runs from 22 to 0, scored with a plain lowercase comparison.
4. With "Detect automatically", the worker runs one decoder step on the first 30 seconds and picks the most likely language token, which is how Whisper itself detects language.

Model downloads (onnx-community conversions):

| Model | WebAssembly (8-bit) | WebGPU (fp32 encoder, 4-bit decoder) |
|---|---|---|
| tiny | 41 MB | 120 MB |
| base | 77 MB | 206 MB |
| small | 249 MB | 586 MB |

## Run it yourself

Serve the folder with any static file server and open `index.html`:

```bash
npx serve .
```

It has to be served over http(s). Opening the file directly will not work, because module workers need an origin.

For multi-threaded WebAssembly, serve the page and the scripts with these two headers. Without them it still works, on one thread:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

Add `?device=wasm` or `?device=webgpu` to the URL to force a backend.

## Privacy

The only network requests are the page itself, transformers.js and the ONNX runtime from jsDelivr, Mediabunny from jsDelivr when a large file needs it, and the model files from huggingface.co and its download CDN. Open the Network tab in your browser's developer tools while you transcribe and you can check that no audio leaves the page.

## Limits

- Browsers without WebCodecs stop at 1 GB on a computer and 200 MB on a phone, and FLAC stops there in every browser. Extract the audio first.
- A file only opens if the browser has a decoder for its audio codec.
- Speed depends on the visitor's hardware. On a processor without WebGPU, a long recording can take longer than the recording itself.
- No speaker labels.

## Desktop version

If you want Whisper for live dictation into any app on Windows or Mac, with larger models on your GPU, that's what [StarWhisper](https://starwhisper.ai/) does. This repo is the browser tool only.

## License

MIT, see [LICENSE](LICENSE). Third-party notices are in [NOTICE](NOTICE).
