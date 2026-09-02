from __future__ import annotations

import unittest

from genivox.system.probe import _parse_nvidia_smi


class SystemProbeTests(unittest.TestCase):
    def test_parses_nvidia_smi_rows_without_crashing_on_bad_memory(self) -> None:
        rows = _parse_nvidia_smi(
            "NVIDIA GeForce RTX 5070 Laptop GPU, 8192, 590.00\n"
            "Experimental GPU, unknown, 591.00\n"
            "malformed\n"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].memory_total_mib, 8192)
        self.assertIsNone(rows[1].memory_total_mib)


if __name__ == "__main__":
    unittest.main()
