"""Route mixed-script text into language-labelled source spans."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass

from genivox.core.models import LanguageSegment

EXPLICIT_LANGUAGES = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "grc",
        "he",
        "hi",
        "id",
        "it",
        "ja",
        "km",
        "ko",
        "la",
        "lo",
        "ms",
        "my",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "th",
        "tl",
        "tr",
        "vi",
        "yue",
        "zh",
    }
)

# Paired BCP-47-like tags allow a custom engine to expose languages beyond the
# built-in presets. The selected adapter remains authoritative and rejects tags
# absent from its manifest.
_TAG_RE = re.compile(r"\[(/?)([a-z]{2,3}(?:-[a-z0-9]{2,8}){0,2})\]", re.IGNORECASE)
_LATIN_WORD_RE = re.compile(
    r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f]+(?:[-'’][A-Za-z\u00c0-\u024f]+)*"
)

_LATIN_FUNCTION_WORDS = frozenset(
    {
        "a",
        "ab",
        "ad",
        "aut",
        "autem",
        "cum",
        "de",
        "dum",
        "e",
        "ego",
        "enim",
        "est",
        "et",
        "ex",
        "hic",
        "hoc",
        "iam",
        "in",
        "ne",
        "nec",
        "non",
        "nos",
        "per",
        "pro",
        "quae",
        "quam",
        "qui",
        "quid",
        "quod",
        "quo",
        "salve",
        "sed",
        "si",
        "sine",
        "sub",
        "sum",
        "sunt",
        "tu",
        "ubi",
        "ut",
        "vale",
        "vos",
    }
)
_LATIN_DIAGNOSTIC_WORDS = frozenset(
    {
        "ab",
        "autem",
        "cum",
        "ego",
        "enim",
        "est",
        "et",
        "ex",
        "hic",
        "hoc",
        "nec",
        "non",
        "quae",
        "quam",
        "qui",
        "quid",
        "quod",
        "quo",
        "salve",
        "sed",
        "sine",
        "sum",
        "sunt",
        "ubi",
        "ut",
        "vale",
        "vos",
    }
)
_LATIN_STRONG_ENDINGS = (
    "arum",
    "orum",
    "ibus",
    "aque",
    "eque",
    "ique",
    "oque",
    "umque",
    "usque",
    "untur",
    "atur",
    "etur",
    "itur",
)
_LATIN_WEAK_ENDINGS = (
    "ae",
    "am",
    "as",
    "em",
    "es",
    "ibus",
    "is",
    "nt",
    "orum",
    "os",
    "um",
    "us",
)


@dataclass(slots=True)
class _Guess:
    language: str
    source: str
    confidence: float


def _script_language(char: str) -> str | None:
    name = unicodedata.name(char, "")
    if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name:
        return "zh"
    if name in {
        "IDEOGRAPHIC ITERATION MARK",
        "IDEOGRAPHIC NUMBER ZERO",
        "VERTICAL IDEOGRAPHIC ITERATION MARK",
    } or name.startswith("HANGZHOU NUMERAL"):
        return "zh"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "ja"
    if "HANGUL" in name:
        return "ko"
    if "GREEK" in name:
        # Script alone cannot distinguish Ancient Greek (grc) from Modern
        # Greek (el); routing it as either would silently choose a pronunciation.
        return "grk"
    if "CYRILLIC" in name:
        return "ru"
    return None


def _script_confidence(language: str) -> float:
    # Script alone cannot distinguish Ancient/Modern Greek, Russian/other Cyrillic,
    # or Chinese/Japanese Han.  The confidence intentionally exposes that ambiguity.
    return {"ja": 0.98, "ko": 0.98, "ru": 0.72, "zh": 0.68}[language]


def _latin_guess(text: str, enabled: bool) -> _Guess:
    """Return a conservative, explicitly uncertain guess for Latin-script text."""

    if not enabled:
        return _Guess("und", "auto", 0.0)

    words = [match.group().casefold() for match in _LATIN_WORD_RE.finditer(text)]
    if not words:
        return _Guess("und", "auto", 0.0)

    function_hits = sum(word in _LATIN_FUNCTION_WORDS for word in words)
    diagnostic_hits = sum(word in _LATIN_DIAGNOSTIC_WORDS for word in words)
    strong_hits = sum(word.endswith(_LATIN_STRONG_ENDINGS) for word in words)
    weak_hits = sum(word.endswith(_LATIN_WEAK_ENDINGS) for word in words)
    evidence = function_hits * 2.0 + strong_hits * 1.5 + weak_hits * 0.35
    density = evidence / len(words)

    # A single familiar-looking word is too ambiguous ("status", "data", etc.).
    if (
        len(words) >= 3
        and evidence >= 2.7
        and density >= 0.55
        and (diagnostic_hits >= 1 or strong_hits >= 1)
    ):
        confidence = min(0.79, 0.50 + density * 0.12 + min(len(words), 12) * 0.01)
        return _Guess("la", "heuristic", round(confidence, 2))

    return _Guess("und", "heuristic", 0.25)


class LanguageRouter:
    """Split one source string into explicit or script-inferred language spans.

    Explicit tags are authoritative and removed from emitted segment text.  ``start``
    and ``end`` always refer to offsets in the original tagged source string.
    """

    def __init__(self, *, latin_heuristic: bool = True) -> None:
        self.latin_heuristic = latin_heuristic

    def segment(self, text: str) -> list[LanguageSegment]:
        if not text:
            return []

        spans = self._explicit_spans(text)
        segments: list[LanguageSegment] = []
        for start, end, language in spans:
            if start == end:
                continue
            if language is not None:
                segments.append(
                    LanguageSegment(
                        text=text[start:end],
                        language=language,
                        start=start,
                        end=end,
                        source="explicit",
                        confidence=1.0,
                    )
                )
            else:
                segments.extend(self._auto_segments(text, start, end))
        merged = self._merge_adjacent(segments)
        # Whitespace between adjacent explicit tags is layout, not a language
        # request. Keeping it as ``und`` creates fake unresolved rows and can
        # make an otherwise explicit mixed-language synthesis fail.
        return [segment for segment in merged if segment.text.strip()]

    @staticmethod
    def _explicit_spans(text: str) -> list[tuple[int, int, str | None]]:
        """Parse flat paired tags; malformed tags remain ordinary source text."""

        tokens = list(_TAG_RE.finditer(text))
        closing_indices: dict[str, deque[int]] = defaultdict(deque)
        for index, token in enumerate(tokens):
            if token.group(1):
                closing_indices[token.group(2).casefold()].append(index)

        spans: list[tuple[int, int, str | None]] = []
        cursor = 0
        token_index = 0
        while token_index < len(tokens):
            opening = tokens[token_index]
            if opening.start() < cursor or opening.group(1):
                token_index += 1
                continue
            language = opening.group(2).casefold()
            candidates = closing_indices[language]
            while candidates and candidates[0] <= token_index:
                candidates.popleft()
            if not candidates:
                token_index += 1
                continue
            closing_index = candidates.popleft()
            closing = tokens[closing_index]
            if cursor < opening.start():
                spans.append((cursor, opening.start(), None))
            spans.append((opening.end(), closing.start(), language))
            cursor = closing.end()
            token_index = closing_index + 1

        if cursor < len(text):
            spans.append((cursor, len(text), None))
        return spans

    def _auto_segments(self, text: str, start: int, end: int) -> list[LanguageSegment]:
        labels: list[_Guess | None] = [None] * (end - start)

        # Japanese normally mixes Han and kana without spaces.  Treat each contiguous
        # Han/kana cluster as Japanese if it contains any kana.
        position = start
        while position < end:
            language = _script_language(text[position])
            if language in {"zh", "ja"}:
                cluster_end = position + 1
                cluster_languages = {language}
                while cluster_end < end:
                    next_language = _script_language(text[cluster_end])
                    if next_language not in {"zh", "ja"}:
                        break
                    cluster_languages.add(next_language)
                    cluster_end += 1
                resolved = "ja" if "ja" in cluster_languages else "zh"
                guess = _Guess(resolved, "auto", _script_confidence(resolved))
                for index in range(position - start, cluster_end - start):
                    labels[index] = guess
                position = cluster_end
                continue
            if language == "grk":
                labels[position - start] = _Guess("und", "greek-script", 0.0)
            elif language is not None:
                labels[position - start] = _Guess(language, "auto", _script_confidence(language))
            position += 1

        latin_matches = list(_LATIN_WORD_RE.finditer(text, start, end))
        latin_groups: list[list[re.Match[str]]] = []
        for match in latin_matches:
            if not latin_groups:
                latin_groups.append([match])
                continue
            gap = text[latin_groups[-1][-1].end() : match.start()]
            if any(
                _script_language(char) is not None
                or unicodedata.category(char).startswith("L")
                for char in gap
            ):
                latin_groups.append([match])
            else:
                latin_groups[-1].append(match)
        for group in latin_groups:
            guess = _latin_guess(
                text[group[0].start() : group[-1].end()], self.latin_heuristic
            )
            for match in group:
                for index in range(match.start() - start, match.end() - start):
                    labels[index] = guess

        # Unsupported alphabets are real ``und`` evidence, not punctuation to absorb
        # into a neighbouring supported language.
        for index, char in enumerate(text[start:end]):
            if labels[index] is None and unicodedata.category(char).startswith("L"):
                labels[index] = _Guess("und", "auto", 0.0)

        # Combining marks inherit the base script before neutral punctuation is filled.
        for index, char in enumerate(text[start:end]):
            if labels[index] is None and unicodedata.category(char).startswith("M") and index:
                labels[index] = labels[index - 1]

        self._fill_neutral(labels)
        return self._segments_from_labels(text, start, end, labels)

    @staticmethod
    def _fill_neutral(labels: list[_Guess | None]) -> None:
        if not labels:
            return
        significant = [index for index, label in enumerate(labels) if label is not None]
        if not significant:
            labels[:] = [_Guess("und", "auto", 0.0)] * len(labels)
            return

        first = significant[0]
        labels[:first] = [labels[first]] * first
        previous = first
        for current in significant[1:]:
            if current > previous + 1:
                # Whitespace and punctuation between scripts stay with the preceding
                # span, making each emitted segment a contiguous original slice.
                labels[previous + 1 : current] = [labels[previous]] * (current - previous - 1)
            previous = current
        labels[previous + 1 :] = [labels[previous]] * (len(labels) - previous - 1)

    @staticmethod
    def _segments_from_labels(
        text: str, start: int, end: int, labels: list[_Guess | None]
    ) -> list[LanguageSegment]:
        if not labels:
            return []
        result: list[LanguageSegment] = []
        run_start = 0
        for index in range(1, len(labels) + 1):
            previous = labels[run_start]
            current = labels[index] if index < len(labels) else None
            if index < len(labels) and current == previous:
                continue
            assert previous is not None
            absolute_start = start + run_start
            absolute_end = start + index
            result.append(
                LanguageSegment(
                    text=text[absolute_start:absolute_end],
                    language=previous.language,
                    start=absolute_start,
                    end=absolute_end,
                    source=previous.source,
                    confidence=previous.confidence,
                )
            )
            run_start = index
        return result

    @staticmethod
    def _merge_adjacent(segments: list[LanguageSegment]) -> list[LanguageSegment]:
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if (
                previous.language == segment.language
                and previous.source == segment.source
                and previous.end == segment.start
            ):
                previous.text += segment.text
                previous.end = segment.end
                previous.confidence = min(previous.confidence, segment.confidence)
            else:
                merged.append(segment)
        return merged


def segment_text(text: str, *, latin_heuristic: bool = True) -> list[LanguageSegment]:
    """Convenience wrapper around :class:`LanguageRouter`."""

    return LanguageRouter(latin_heuristic=latin_heuristic).segment(text)
