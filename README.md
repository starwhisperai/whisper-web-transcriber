# Whisper Web Transcriber

Transcribe audio and video files with OpenAI's Whisper inside the browser. The file is decoded and transcribed on the visitor's own machine. Nothing is uploaded.

Live version: https://starwhisper.ai/whisper-web.html

- Runs Whisper tiny, base or small through [transformers.js](https://github.com/huggingface/transformers.js)
- Uses WebGPU when the browser has it, WebAssembly otherwise
- Detects the spoken language, or you pick it; can translate into English
- Exports plain text, SRT and VTT
- No build step, no server code, no dependencies to install

## How it works

1. `src/app.js` reads the file with `File.arrayBuffer()` and decodes it with an `OfflineAudioContext` at 16 kHz, mixed down to mono.
2. The samples are transferred to `src/worker.js`, a module worker that loads transformers.js from jsDelivr and the ONNX model from the Hugging Face hub. The browser caches both after the first run.
3. Long recordings are cut into 30 second windows with 5 seconds of overlap on each side, and transformers.js merges the overlapping text.
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

The only network requests are the page itself, transformers.js and the ONNX runtime from jsDelivr, and the model files from huggingface.co and its download CDN. Open the Network tab in your browser's developer tools while you transcribe and you can check that no audio leaves the page.

## Limits

- The whole file is read into memory before decoding, so very large video files can fail. Extract the audio first.
- Speed depends on the visitor's hardware. On a processor without WebGPU, a long recording can take longer than the recording itself.
- No speaker labels.

## Desktop version

If you want Whisper for live dictation into any app on Windows or Mac, with larger models on your GPU, that's what [StarWhisper](https://starwhisper.ai/) does. This repo is the browser tool only.

## License

MIT, see [LICENSE](LICENSE). Third-party notices are in [NOTICE](NOTICE).
