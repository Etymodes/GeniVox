from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from genivox.experiments import ExperimentRecord, ExperimentStore, HumanRating


class ExperimentStoreTests(unittest.TestCase):
    def test_append_read_and_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ExperimentStore(Path(temporary_directory) / "experiments.jsonl")
            record = ExperimentRecord(
                engine_id="mock",
                text="Salve, κόσμε.",
                audio_path="output.wav",
                parameters={"seed": 7},
            )
            store.append(record)
            self.assertEqual(store.read_all()[0].text, record.text)

            store.update_rating(record.id, HumanRating(naturalness=4, pronunciation=3))
            updated = store.read_all()[0]
            self.assertEqual(updated.rating.naturalness, 4)

    def test_rating_range_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            HumanRating(naturalness=6).validate()


if __name__ == "__main__":
    unittest.main()
