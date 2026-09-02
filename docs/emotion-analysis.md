# Emotion analysis: optional emotion2vec bridge

GeniVox treats pitch, loudness and an RMS-envelope peak-rate proxy as **acoustic features**, not emotion
labels. The peak rate is not a syllable/word speaking-rate measurement and v0.1 does not automatically
map it back to the synthesis speed control. When you explicitly enable the optional bridge below, a local FunASR emotion2vec model can
add model-inferred emotion probabilities. Those probabilities are uncertain estimates rather than
facts about a speaker's internal state. This bridge only analyzes audio; applying its output requires a
separately configured process bridge that declares and actually consumes `emotion_vector`.

## Isolated local environment

Keep this model in a separate Python environment so its PyTorch and CUDA requirements do not conflict
with GPT-SoVITS or other engines. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-emotion2vec
& .\.venv-emotion2vec\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-emotion2vec\Scripts\python.exe -m pip install funasr modelscope
```

This installation command installs Python packages only. GeniVox does not install FunASR or fetch
emotion2vec weights automatically. The first bridge invocation may download the selected model from
`--hub`; that happens only when you run the bridge. For an offline system, pre-stage a model using the
upstream tooling and pass its local directory to `--model`. Model weights retain their upstream license.

## JSON bridge contract

The bridge reads exactly one request object from standard input:

```json
{"audio_path":"C:\\voices\\reference.wav"}
```

Example invocation:

```powershell
$request = '{"audio_path":"C:\\voices\\reference.wav"}'
$request | & .\.venv-emotion2vec\Scripts\python.exe `
  .\scripts\bridges\emotion2vec_bridge.py `
  --model iic/emotion2vec_plus_large --device cuda --hub ms
```

Successful standard output is one compact machine-readable JSON object:

```json
{"emotion":{"angry":0.02,"neutral":0.91},"meta":{"unknown_labels":[]}}
```

Third-party progress and diagnostics are redirected to standard error. A missing FunASR installation,
bad request, missing audio file or incompatible result exits non-zero. The bridge recognizes the common
FunASR direct-dictionary and list-of-dictionaries `labels`/`scores` results. Known bilingual labels such
as `生气/angry` are normalized; unfamiliar labels remain visible as `unknown:<original>` and are also
listed in `meta.unknown_labels` rather than being silently discarded.

The command can be connected to `genivox.audio.ExternalEmotionAnalyzer`:

```python
from genivox.audio import ExternalEmotionAnalyzer

analyzer = ExternalEmotionAnalyzer(
    command=[
        r".venv-emotion2vec\Scripts\python.exe",
        r"scripts\bridges\emotion2vec_bridge.py",
        "--device",
        "cuda",
    ],
    timeout_seconds=180,
)
```

To let the desktop controller use the bridge, set its argument array as JSON before launching GeniVox:

```powershell
$emotionCommand = @(
  (Resolve-Path .\.venv-emotion2vec\Scripts\python.exe).Path,
  (Resolve-Path .\scripts\bridges\emotion2vec_bridge.py).Path,
  "--device", "cuda"
)
$env:GENIVOX_EMOTION_COMMAND_JSON = ConvertTo-Json $emotionCommand -Compress
.\scripts\run_windows.ps1
```

The current bridge is intentionally one-shot and favors dependency isolation over latency. A persistent
worker can be added later to keep model weights resident in GPU memory after the resource behavior has
been measured on target hardware.
