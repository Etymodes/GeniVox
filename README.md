# GeniVox

GeniVox v0.1 is a local-first desktop workbench prototype for prosody inspection, multilingual text
routing, model registration, externally managed training runs, and controlled TTS experiments.

The project does **not** merge incompatible checkpoints into one neural network. Instead, it gives
GPT-SoVITS, IndexTTS, VoxCPM and future engines a common desktop workflow while each engine runs in
its own Python environment. The current single-sample voice profile stores an authorized source path,
hash, transcript and analysis; loading/relinking profile libraries and attaching trained artifacts are
planned rather than implemented in v0.1.

## Current v0.1 scope

- PySide6/Qt desktop interface for synthesis, voice analysis, multilingual routing, training and experiments.
- Capability-aware engine manifests, a functional GPT-SoVITS HTTP adapter, and a documented generic
  process-bridge contract. IndexTTS and VoxCPM currently have registration presets only.
- Mixed-script segmentation plus explicit tags such as `[la]`, `[grc]`, `[el]`, `[ru]`, `[en]` and `[zh]`.
- Offline WAV quality/prosody analysis; emotion recognition is an optional local analyzer, never guessed.
- Read-only dataset audit and distribution guidance; supervision of user-supplied external training
  commands with logs and JSONL metric charts.
- A default workspace outside the repository. A custom workspace inside a Git repository is not
  automatically protected from version control.

Heavy model weights are intentionally not downloaded during installation. Correct Classical Latin and
Ancient Greek synthesis also requires a selected historical pronunciation and a compatible phoneme-aware
engine or a fine-tuned model; v0.1 records the choice as metadata but does not implement a scholarly G2P.

## Launch the workbench on Windows 11

Requirements: 64-bit Python 3.11, Git, and Windows 10/11. No GPU is needed for the UI or Mock WAV test.
A GPU is strongly recommended for most neural backends, but CPU/offload support depends on that backend.

```powershell
git clone --branch feature/desktop-workbench-v0.1 --single-branch https://github.com/Etymodes/GeniVox.git
Set-Location GeniVox
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

This starts the desktop UI and its deterministic Mock engine. It does **not** install GPT-SoVITS,
download weights, or train a voice. For real GPT-SoVITS output, install an official compatible release,
start its local `api_v2.py`, then register its loopback `/tts` endpoint in **模型管理**; see
[Local model integration](docs/model-integration.md). On an RTX 5070 Laptop GPU with 8 GB VRAM, begin
with inference and small GPT-SoVITS experiments. VoxCPM2's documented local LoRA/full-training budgets
are about 20/40 GB, so those training paths need larger or remote hardware.

PCM WAV analysis works in the base installation. For optional Latin, Ancient Greek and Russian IPA
previews, install [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) and either add it to
`PATH` or set `GENIVOX_ESPEAK_PATH` to its executable.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Architecture and integration details:

- [Product specification](docs/product-spec.md)
- [Architecture](docs/architecture.md)
- [Verified engine capability matrix](docs/engine-matrix.md)
- [Model integration](docs/model-integration.md)
- [Local process bridge contract](docs/process-bridge.md)
- [Classical and mixed-language strategy](docs/language-strategy.md)
- [Optional emotion2vec analysis](docs/emotion-analysis.md)

## Privacy and responsible use

Recordings, model references and datasets stay under `%USERPROFILE%\GeniVoxWorkspace` by default,
outside the cloned repository. Do not point `GENIVOX_WORKSPACE` into a tracked directory. On POSIX
systems, GeniVox narrows its workspace directories to owner-only (`0700`) and private
JSON/log files to `0600`; on Windows, a custom workspace inherits that directory's existing ACL, so do
not choose a shared folder. Voice profiles and experiment ledgers deliberately retain source paths,
transcripts, text and parameters for reproducibility. They are local, but they are still sensitive data.

The built-in GPT-SoVITS adapter only connects to a loopback address and ignores system HTTP proxies and
redirects. A process bridge or training command is different: after explicit confirmation, it runs as
your current OS user and is **not a sandbox**. Review `genivox-engine.json` and the referenced code before
trusting it; avoid putting credentials in command-line arguments.

Only use a person's voice with informed authorization. Generated speech should be disclosed where
listeners could mistake it for a real recording. GeniVox must not be used for impersonation, fraud,
harassment, bypassing authentication, or deceptive evidence.

GeniVox source code is MIT licensed. Every upstream model, checkpoint and dataset keeps its own license;
registering it in GeniVox does not change those terms.
