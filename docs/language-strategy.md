# Classical and mixed-language strategy

## One text box, multiple languages

The router detects unambiguous scripts and preserves their order. Explicit markup resolves ambiguous
Latin-script passages:

```text
[la]Gallia est omnis divisa in partes tres.[/la]
[grc]Ἄνδρα μοι ἔννεπε, Μοῦσα.[/grc]
[el]Η σύγχρονη ελληνική πρόταση.[/el]
[ru]В начале было слово.[/ru]
[en]Then the English commentary continues.[/en]
```

Greek script is detected but remains unresolved until the user selects Ancient Greek `[grc]` or Modern
Greek `[el]`. Cyrillic does not uniquely identify Russian, and Han alone does not distinguish Chinese
from Japanese; their low-confidence guesses require `[ru]`, `[zh]`, `[ja]` or a selected fallback before
synthesis. Kana and Hangul are routed with higher confidence. Latin, English,
romanized Greek and names cannot be reliably distinguished from Unicode alone, so heuristic guesses
remain marked as uncertain and must be confirmed before a historical pronunciation is dispatched.
“Automatic” never silently chooses a historical pronunciation.

## Pronunciation presets

Initial defaults are working assumptions, not claims of a single correct standard:

- Latin: reconstructed Classical Latin, roughly late Republican/early Imperial educated pronunciation.
- Ancient Greek: reconstructed fifth–fourth century BCE Attic.
- Russian: contemporary standard Russian.

The v0.1 selectors are stored as experiment metadata but do not change eSpeak's IPA. Scholarly rule sets
for Ecclesiastical Latin, Koine, Erasmian and other reconstructions are future work.

## Frontend path

```text
text span → normalization → language-specific G2P → editable phonemes → engine adapter → audio
```

An optional eSpeak-ng frontend provides a reproducible baseline for `la`, `grc` and `ru`. Its output is
not automatically accepted as scholarly reconstruction: the IPA column is human-editable and is passed
only to a process bridge that declares `phoneme_input`. The built-in GPT-SoVITS HTTP adapter ignores this
plan; classical profiles still need explicit scholarly rule sets and tests.

## Preserving one voice across a mixed passage

The same registered engine configuration, checkpoint path and voice reference are held constant while
target language IDs/phonemes change for each span. Outputs are assembled in source order. A process bridge
is invoked once per segment, so v0.1 does not guarantee one resident model, loudness matching, cross-fade
or prosodic continuity. If the selected engine lacks a language or phoneme-input capability, the job stops
with an explanation instead of substituting another language.

Russian is supported by several modern multilingual engines. Classical Latin and Ancient Greek generally
are not native pretrained targets; high-quality results will require phoneme-conditioned synthesis or a
carefully documented multilingual fine-tune using licensed, pronunciation-consistent data.
