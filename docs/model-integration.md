# Local model integration

## Workspace layout

By default GeniVox creates `%USERPROFILE%\GeniVoxWorkspace`:

```text
GeniVoxWorkspace/
  engines/       engine manifest JSON files
  models/        checkpoints or links to checkpoint directories
  datasets/      private source and processed data
  profiles/      portable voice-profile manifests
  runs/          configs, logs and metrics
  outputs/       synthesized audio and experiment metadata
```

Set `GENIVOX_WORKSPACE` to use another drive. Large model repositories may stay elsewhere; a manifest
can point to their absolute path and their own Python executable.

On POSIX, workspace directories and private manifests/logs are tightened to owner-only permissions. On
Windows, a workspace on another drive or network share inherits that location's ACL; use a directory only
your account can read. Engine processes are explicitly trusted local code, not sandboxes.

## GPT-SoVITS `api_v2.py`

Recommended first integration because it has a mature Windows UI/toolchain and lightweight few-shot
fine-tuning. Install a compatible official release and weights using its own instructions, start
`api_v2.py`, then use **模型管理 → 登记模型** with transport `HTTP` and the loopback endpoint
`http://127.0.0.1:9880/tts`. The built-in adapter maps reference audio, exact reference transcript,
prompt/target language, speed and seed. GPT-SoVITS does not receive a numeric emotion vector or free-form style instruction;
those controls must not be shown as successfully applied. Expression comes mainly from reference audio
or an engine-specific fine-tuned preset.

### Start the local service with GeniVox

Once the upstream installation, Python environment and weights work when launched manually, GeniVox
can start its HTTP service on each app launch. In **模型管理**, select the registered GPT-SoVITS HTTP
service (including the built-in `gpt-sovits-v2-local` row), choose **配置已选服务**, set its source
directory and its own Python executable, enable **允许启动此目录中的本地模型代码** and
**随 GeniVox 启动本地 GPT-SoVITS 服务**, then save. This is opt-in and does not install or
download the upstream model. The configured endpoint must be `http://127.0.0.1:<port>/tts`.

GeniVox launches the upstream `api_v2.py -a 127.0.0.1 -p <port>` from the selected source
directory with that Python executable, in a background worker, and waits for the read-only
`/openapi.json` probe. This follows the [upstream API invocation](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py).
The initial model load can take time. Startup failure and the local log path appear in the model
manager. A compatible service that is already running is reused; a different service occupying
the port is never replaced. On exit GeniVox stops only the process it started, leaving a manually
launched service alone. It never changes active checkpoint weights as part of startup.

The desktop accepts WAV, FLAC, MP3, M4A and OGG reference recordings. It decodes them locally to
mono PCM16 WAV before the API request, keeps the temporary file available until synthesis completes,
then removes it. The original recording remains unchanged; the reference must be at most 30 seconds.
For GPT-SoVITS, use a clean 3–10 second clip as required by the upstream inference path.

The latest release verified for this document is `20250606v2pro`; the adapter targets the public
`api_v2.py` request contract and is not pinned to a tested upstream commit. Its public language set does not include Russian,
Latin or Ancient Greek. V1/V2/V3/V4/V2Pro/ProPlus components are version-specific; preserve the complete
S1/S2/vocoder family and revision instead of mixing files that happen to share an extension.

### Read-only API probe and acceptance levels

The `v2` in `api_v2.py` identifies the HTTP API contract, not the loaded GPT-SoVITS model generation.
One `api_v2.py` service may load v1, v2, v3, v4, v2Pro or v2ProPlus components. The service does not
provide a read-only endpoint that reports its active GPT checkpoint, SoVITS checkpoint, device or model
generation. In particular, do not interpret the OpenAPI document's `info.version` as a model version.

After starting the service from the GPT-SoVITS repository root, this PowerShell check performs one
read-only `GET /openapi.json` request and verifies that the advertised API contains `POST /tts`:

```powershell
$openapi = curl.exe --noproxy "*" --max-time 3 --fail --silent --show-error `
  http://127.0.0.1:9880/openapi.json | ConvertFrom-Json
$ttsRoute = $openapi.paths | Select-Object -ExpandProperty '/tts'
if ($null -eq $ttsRoute.post) {
    throw 'The local service does not advertise the GPT-SoVITS POST /tts contract.'
}
'ROUTE_PRESENT: POST /tts is advertised; use GeniVox to validate its request shape.'
```

Keep the API bound to `127.0.0.1`. The official API has unauthenticated process-control and
model-switching routes, so a health check must never call `/control`, `/set_gpt_weights`,
`/set_sovits_weights` or `/set_refer_audio`. It must also avoid redirects and system proxies.

Interpret the read-only probe states and the subsequent acceptance level as follows:

- `INVALID`: the configured endpoint is absent, malformed, non-loopback or contains credentials,
  a query or a fragment.
- `OFFLINE`: the loopback service did not answer within the bounded timeout.
- `WRONG_SERVICE`: something answered on the port, but it did not return a usable GPT-SoVITS OpenAPI
  document with a `/tts` route.
- `INCOMPATIBLE`: an HTTP service answered, but its OpenAPI document did not advertise the expected
  `POST /tts` contract.
- `API_READY`: the OpenAPI 3 document advertises `POST /tts`, its core text fields accept strings,
  and it has no unknown mandatory field that the adapter cannot send. This is a request-shape match,
  not proof of service identity, model generation or successful synthesis with a particular reference
  recording and language.
- `SYNTHESIS_VERIFIED` is a later acceptance level, not a result of the read-only probe: a separately
  requested, non-streaming `POST /tts` completed with an authorized reference recording and returned
  a non-empty, complete PCM WAV that passed validation.

The repository's automated tests use mocked HTTP responses and deterministic test audio. No official
GPT-SoVITS weight files are included. On 2026-09-30, a user-side GPT-SoVITS run with an M4A reference
produced a non-silent 4.46-second PCM WAV; this validates a real local output path beyond the mocked
tests. The upstream revision, selected weights, GPU/CUDA/PyTorch stack, and audible quality still need
to be recorded for complete device acceptance. Until then, display the model generation as unverified
even when the API is ready.

## IndexTTS 2.5

Use a separate Python 3.10/3.11 environment following the upstream repository. Upstream version 2.5 can
separate speaker identity from an emotion reference, an eight-component vector or emotion text and offers
a duration factor. GeniVox v0.1 does not include an IndexTTS bridge: the preset only fills manifest
capabilities. A user-reviewed process bridge must explicitly map and verify these fields.

IndexTTS checkpoints are not compatible with GPT-SoVITS checkpoints. Reuse the verified source dataset,
not weight files. Check the checkpoint license independently from the GeniVox MIT license.

The official 2.5 repository/model card currently lists Chinese, English, Japanese, Spanish and Arabic,
and does not publish an official fine-tuning/LoRA workflow. GeniVox therefore treats the built-in
IndexTTS 2.5 preset as inference-only unless a separately reviewed bridge explicitly declares a training
command. Its Bilibili Model Use License must be reviewed independently.

## VoxCPM2

Use an isolated environment and its native controllable-cloning entry point. VoxCPM 2.0.3 supports
Russian and modern Greek among its 30 advertised languages, but not Latin or Ancient Greek. GeniVox v0.1
does not ship a VoxCPM bridge; a future or user-supplied bridge should map the same voice reference plus
a natural-language instruction such as “measured, low-energy delivery, slightly faster.”
Instructions are stochastic and should be stored with seed and output for comparison. About 8 GB of VRAM
is a tight boundary for inference on a display GPU. The upstream fine-tuning recipe documents roughly
20 GB for LoRA and 40 GB for full fine-tuning, but actual use is hardware, sequence-length and software
dependent. Quantization, offload or remote training must be validated rather than assumed.
VoxCPM 2.x LoRA artifacts must record the exact base revision, rank and configuration.

## Import versus installation

In v0.1, “Import” registers existing paths and reads an explicitly supplied `genivox-engine.json`; it does
not probe model code, calculate checkpoint hashes or copy weights. A future verified import flow will probe
versions/capabilities and record hashes. “Install” will later be an opt-in recipe that
clones the official upstream, creates its environment and downloads user-selected weights after displaying
their license and expected disk/VRAM requirements.
