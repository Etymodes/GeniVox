from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from genivox.languages import (
    EspeakNgPhonemizer,
    LanguageRouter,
    PhonemizerUnavailableError,
    UnsupportedLanguageError,
    segment_text,
)


class LanguageRouterTests(TestCase):
    def test_explicit_tags_override_script_and_preserve_source_indices(self) -> None:
        text = "前言 [la]arma virumque cano[/la] 后记 [grc]μῆνιν ἄειδε[/grc]"

        segments = segment_text(text)

        latin = next(
            segment
            for segment in segments
            if segment.source == "explicit" and segment.language == "la"
        )
        greek = next(
            segment
            for segment in segments
            if segment.source == "explicit" and segment.language == "grc"
        )
        self.assertEqual(latin.text, "arma virumque cano")
        self.assertEqual(text[latin.start : latin.end], latin.text)
        self.assertEqual(latin.confidence, 1.0)
        self.assertEqual(greek.text, "μῆνιν ἄειδε")
        self.assertEqual(text[greek.start : greek.end], greek.text)

    def test_every_explicit_language_tag_is_supported(self) -> None:
        for language in ("la", "grc", "el", "ru", "en", "zh", "ja", "ko", "yue", "fr"):
            with self.subTest(language=language):
                [segment] = segment_text(f"[{language}]sample[/{language}]")
                self.assertEqual(segment.language, language)
                self.assertEqual(segment.text, "sample")
                self.assertEqual(segment.source, "explicit")

    def test_custom_engine_language_can_use_a_generic_iso_tag(self) -> None:
        [segment] = segment_text("[uk]Привіт, світе![/uk]")

        self.assertEqual(segment.language, "uk")
        self.assertEqual(segment.text, "Привіт, світе!")
        self.assertEqual(segment.source, "explicit")

    def test_whitespace_between_explicit_tags_is_not_an_unresolved_segment(self) -> None:
        segments = segment_text("[la]salve[/la]   [grc]χαῖρε[/grc]\n[ru]привет[/ru]")

        self.assertEqual([segment.language for segment in segments], ["la", "grc", "ru"])
        self.assertEqual([segment.text for segment in segments], ["salve", "χαῖρε", "привет"])

    def test_greek_ambiguity_is_not_erased_when_adjacent_text_is_also_und(self) -> None:
        segments = segment_text("χαῖρε hello")

        self.assertEqual([segment.language for segment in segments], ["und", "und"])
        self.assertEqual(
            [segment.source for segment in segments], ["greek-script", "heuristic"]
        )


    def test_auto_routes_mixed_scripts_and_merges_japanese_han_kana(self) -> None:
        text = "λόγος Привет 世界 今日は晴れ 한글"

        segments = LanguageRouter(latin_heuristic=False).segment(text)

        self.assertEqual(
            [(item.language, item.text.rstrip()) for item in segments],
            [
                ("und", "λόγος"),
                ("ru", "Привет"),
                ("zh", "世界"),
                ("ja", "今日は晴れ"),
                ("ko", "한글"),
            ],
        )
        self.assertTrue(all(text[item.start : item.end] == item.text for item in segments))


    def test_latin_letters_default_to_und_when_evidence_is_weak(self) -> None:
        [segment] = segment_text("Hello world from GeniVox")

        self.assertEqual(segment.language, "und")
        self.assertEqual(segment.source, "heuristic")
        self.assertLess(0.0, segment.confidence)
        self.assertLess(segment.confidence, 0.5)

        [ordinary_english] = segment_text("This is a test")
        self.assertEqual(ordinary_english.language, "und")


    def test_classical_latin_heuristic_is_labelled_as_uncertain(self) -> None:
        [segment] = segment_text("Arma virumque cano, Troiae qui primus ab oris.")

        self.assertEqual(segment.language, "la")
        self.assertEqual(segment.source, "heuristic")
        self.assertLessEqual(0.5, segment.confidence)
        self.assertLess(segment.confidence, 0.8)

        [greeting] = segment_text("Salve amice, quid agis?")
        self.assertEqual(greeting.language, "la")
        self.assertEqual(greeting.source, "heuristic")


    def test_explicit_language_is_needed_for_ambiguous_latin_script(self) -> None:
        [auto] = segment_text("Gallia")
        [explicit] = segment_text("[la]Gallia[/la]")

        self.assertEqual(auto.language, "und")
        self.assertEqual(explicit.language, "la")
        self.assertEqual(explicit.source, "explicit")
        self.assertEqual(explicit.start, len("[la]"))

    def test_unsupported_script_is_not_absorbed_by_latin_heuristic(self) -> None:
        text = "hello مرحبا salve quid agis"

        segments = segment_text(text)

        self.assertEqual(segments[0].language, "und")
        self.assertTrue(segments[0].text.startswith("hello "))
        arabic = next(segment for segment in segments if "مرحبا" in segment.text)
        self.assertEqual(arabic.language, "und")
        self.assertEqual(text[arabic.start : arabic.end], arabic.text)
        self.assertEqual(segments[-1].language, "la")

    def test_unicode_script_names_cover_supplementary_blocks_without_coptic_false_positive(
        self,
    ) -> None:
        samples = {
            "Ꙁ": "ru",
            "豈": "zh",
            "〇": "zh",
            "〡": "zh",
            "〻": "zh",
            "𛀀": "ja",
            "ﾡ": "ko",
            "𐅀": "und",
            "Ϣ": "und",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                [segment] = segment_text(text)
                self.assertEqual(segment.language, expected)

    def test_adjacent_same_language_runs_merge(self) -> None:
        text = "λόγος, κόσμος"

        [segment] = segment_text(text)

        self.assertEqual(segment.language, "und")
        self.assertEqual(segment.source, "greek-script")
        self.assertEqual(segment.text, text)
        self.assertEqual((segment.start, segment.end), (0, len(text)))


class EspeakNgPhonemizerTests(TestCase):
    def test_espeak_frontend_does_not_require_espeak_at_import_time(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing = EspeakNgPhonemizer(Path(temporary_directory) / "missing-espeak-ng")

            self.assertFalse(missing.available)
            self.assertTrue(missing.supports("la"))
            with self.assertRaises(PhonemizerUnavailableError):
                missing.phonemize("salve", "la")
            with self.assertRaises(UnsupportedLanguageError):
                missing.phonemize("hello", "en")

    def test_espeak_frontend_maps_supported_languages_and_uses_stdin(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "espeak-ng"
            executable.touch(mode=0o700)
            frontend = EspeakNgPhonemizer(executable)

            for language in ("la", "grc", "ru"):
                with self.subTest(language=language), patch(
                    "genivox.languages.phonemizer.subprocess.run"
                ) as run:
                    run.return_value.stdout = "ipa\n"
                    self.assertEqual(frontend.phonemize("source text", language), "ipa")
                    args, kwargs = run.call_args
                    self.assertEqual(args[0][-3:], ["-v", language, "--stdin"])
                    self.assertEqual(kwargs["input"], "source text")
                    self.assertTrue(Path(args[0][0]).is_absolute())


if __name__ == "__main__":
    main()
