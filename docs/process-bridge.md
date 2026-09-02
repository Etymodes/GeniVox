# Local process bridge contract

A third-party model remains in its own repository and Python environment. Put a
`genivox-engine.json` file in that repository root, then choose the root and interpreter in **模型管理**.
GeniVox reads the manifest when the user presses **登记模型**; registration does not execute code.

See [`examples/engines/genivox-engine.example.json`](../examples/engines/genivox-engine.example.json).

## Synthesis command

For each request, GeniVox runs the configured `command` with the registered interpreter and root as its
working directory. It writes one UTF-8 JSON object to standard input:

```json
{
  "text": "Salve",
  "output_path": "C:\\GeniVoxWorkspace\\outputs\\speech.wav",
  "engine_id": "classics-local",
  "language": "la",
  "segments": [],
  "reference_audio": "C:\\voices\\me.wav",
  "prompt_text": "",
  "speed": 1.0,
  "emotion": {},
  "style_instruction": "",
  "seed": 42,
  "extra": {}
}
```

The bridge writes only one result object to standard output and puts progress diagnostics on standard
error:

```json
{"ok":true,"output_path":"C:\\GeniVoxWorkspace\\outputs\\speech.wav","duration_seconds":2.4,"metadata":{}}
```

The returned `output_path` must resolve to the exact destination requested by GeniVox, and the output
must currently be uncompressed PCM WAV. A non-zero exit, invalid JSON, missing file or
unsupported language becomes a visible failed job; GeniVox never substitutes another engine.

## Training command

`training_command` is an argument array, not a shell string. The controller replaces placeholders such
as `{dataset_path}`, `{output_path}`, `{epochs}`, `{batch_size}` and `{learning_rate}` from the reviewed
UI configuration, then starts the process with `shell=False`. A UI setting affects training **only if**
the manifest command explicitly references its placeholder; there is no universal training-config API.
Standard output is persisted as a log.
A line that is itself JSON can update the chart, for example:

```json
{"step":120,"loss":1.42,"validation_loss":1.67}
```

Cancellation terminates the process group. Pause/resume is not part of the v0.1 portable contract. An
exit code of zero marks the external command successful; v0.1 does not verify, register or load a newly
created checkpoint, so the bridge/training script must validate its own artifacts.

## Trust boundary

Registering metadata does not execute the bridge. Synthesis or training does execute it after the user
checks the local-code trust confirmation. That confirmation is an execution gate, not isolation: the
process has the current user's filesystem permissions and inherits its normal runtime environment. Audit
the repository again after upgrades or pulls, then remove and re-register it if its code changed. Keep
tokens out of command arguments; GeniVox redacts conventional secret flags from saved run manifests, but
process listings and third-party logs are outside its control.
