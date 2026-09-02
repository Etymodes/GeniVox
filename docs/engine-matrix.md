# TTS engine matrix (verified 2026-09-02)

This matrix separates capabilities advertised by official projects from GeniVox adapter support.
An imported path is not proof that a model can perform every listed task.

| Engine | Public release | Official language boundary | Upstream control/training | GeniVox v0.1 status |
|---|---|---|---|---|
| GPT-SoVITS | [20250606v2pro](https://github.com/RVC-Boss/GPT-SoVITS/releases/tag/20250606v2pro) | Chinese, English, Japanese, Korean and Cantonese; no official Russian, Latin or Ancient Greek target | Numeric `speed_factor`; expression mainly follows reference; official training WebUI | Built-in loopback HTTP adapter for the public `api_v2.py` `/tts` contract; real inference needs a separately installed service and weights |
| IndexTTS | [v2.5.0](https://github.com/index-tts/index-tts/releases/tag/v2.5.0) | Chinese, English, Japanese, Spanish and Arabic; no official Russian, Latin or Ancient Greek target | Emotion reference, fixed 8-value vector/text and `duration_factor`; no official 2.5 fine-tuning recipe | Registration preset only; no bundled bridge or end-to-end inference validation |
| VoxCPM | [2.0.3](https://github.com/OpenBMB/VoxCPM/releases/tag/2.0.3) | 30 languages including Russian and modern Greek (`el`); not Latin (`la`) or Ancient Greek (`grc`) | Natural-language delivery instruction; official full SFT and LoRA | Registration preset only; no bundled bridge or end-to-end inference/training validation |

Sources: the official [GPT-SoVITS repository](https://github.com/RVC-Boss/GPT-SoVITS),
[IndexTTS repository](https://github.com/index-tts/index-tts), and
[VoxCPM documentation](https://voxcpm.readthedocs.io/en/latest/).

## What “one voice, one mixed passage” means

GeniVox keeps one registered engine configuration, checkpoint reference and speaker reference fixed,
segments the text, changes the declared language/pronunciation frontend, then joins PCM output. It does
not silently substitute another model. The generic process adapter starts a new process for each segment,
so this does not guarantee one resident model instance, consistent timbre, matched loudness or continuous
cross-segment prosody. It works only when a separately implemented bridge supports every segment.

VoxCPM2 is the strongest of these three candidates for Russian. Its official `Greek` result is modern
Greek, not reconstructed Ancient Greek. GPT-SoVITS and IndexTTS cannot honestly be labelled Russian,
Classical Latin or Ancient Greek engines using their public checkpoints.

## Classical Latin and Ancient Greek

The official [eSpeak NG language table](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
contains `la`, `grc` and `ru`, so eSpeak can supply a reproducible pronunciation preview or dataset-check
frontend. The three neural engines above do not accept arbitrary eSpeak/IPA sequences as a universal
input, however. High-quality historical-language output therefore needs one of:

1. a phoneme-aware model whose vocabulary includes the selected reconstruction;
2. a new, pronunciation-consistent multilingual training corpus and compatible base model; or
3. an explicitly experimental eSpeak-to-voice-conversion fallback, with the expected mechanical
   prosody documented.

Personal voice recordings can adapt timbre. They do not by themselves teach a base model a new language.
VoxCPM's upstream fine-tuning recipe gives roughly 20 GB VRAM for LoRA and 40 GB for full fine-tuning;
actual use varies with hardware, sequence length and software revisions. An 8 GB laptop GPU should use
inference/offload experiments or remote training rather than promise local VoxCPM2 training.

## Licenses and checkpoint ABI

- GPT-SoVITS code and official weights are MIT-licensed.
- VoxCPM is Apache-2.0.
- IndexTTS uses the Bilibili Model Use License, not MIT/Apache; review it before redistribution or
  commercial deployment.
- No official converter makes checkpoints interchangeable among these architectures. A portable
  GeniVox manifest format can reference recordings and engine artifacts, but v0.1's UI does not yet load,
  relink or package those files and never converts neural tensors between architectures.
