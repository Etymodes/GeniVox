# GeniVox v0.1 product specification

## Goal

Provide one local desktop application in which a user can:

1. register separately installed TTS engines and local checkpoints;
2. analyze an authorized reference recording without uploading it;
3. synthesize mixed-language text in one job while preserving one voice identity;
4. audit a fine-tuning dataset before training;
5. launch and observe engine-native training commands;
6. compare outputs, parameters and human listening preferences reproducibly.

## Non-goals for v0.1

- Converting a checkpoint between unrelated architectures. A GPT-SoVITS `.pth` cannot become an
  IndexTTS or VoxCPM checkpoint without retraining.
- Claiming emotion from pitch or loudness alone. Acoustic proxies are shown separately from results of
  an actual speech-emotion-recognition model.
- Silently pretending that an engine supports an unsupported language or control.
- Automatically downloading, accepting licenses for, or executing third-party model repositories.
- Training a universal Ancient Greek or Latin pronunciation. Historical pronunciation is a user choice.

## Verifiable acceptance criteria

| Area | v0.1 criterion |
|---|---|
| Desktop | The application opens with no model installed and exposes all workbench pages. |
| Engine registry | A valid manifest reloads; invalid capability/transport values produce a useful error. |
| Synthesis | The mock engine creates a valid WAV; the GPT-SoVITS adapter builds the documented API request. |
| Mixed language | Explicit tags are lossless and runs preserve source order. Greek, low-confidence Cyrillic and pure-Han spans require an explicit language tag or selected fallback before dispatch. |
| Voice analysis | A PCM WAV produces duration, level, clipping/silence and F0-related measurements. |
| Emotion | No model means “not analyzed,” never an invented emotion label. |
| Dataset | Missing/invalid WAV files, duplicates and duration/language/emotion imbalance are reported without mutation. Other codecs are not decoded by the base audit. |
| Training | A configured child process streams logs/metrics and can be cancelled without blocking the UI. |
| Privacy | The default workspace is outside the repository; POSIX private paths receive owner-only modes. A custom path still needs the user's own ACL and Git policy. |

## Iteration plan

### v0.1 — workbench spine

Qt UI, local workspace, engine contracts, GPT-SoVITS HTTP adapter, mixed-text routing, acoustic analysis,
dataset audit, training runner and testable mock synthesis.

### v0.2 — local model bridges

Install/import assistants for GPT-SoVITS v2ProPlus, IndexTTS 2.5 and VoxCPM2; guided emotion2vec setup and
an ASR bridge; microphone recording; output stitching with loudness matching and cross-fade.

### v0.3 — portable voice passport

Consent/provenance metadata, verified transcripts, pronunciation layers, train/validation split guidance,
per-engine fine-tuning recipes and automatic baseline-versus-fine-tune evaluations.

### v0.4 — classical-language experiments

Restored Classical Latin and reconstructed Attic/Koine profiles, human-correctable phoneme editor,
alignment visualizer and controlled multilingual fine-tuning experiments.
