#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from kx import parse_kx  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


class PartyKxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = parse_kx((ROOT / "assets/debris_party.kx").read_bytes())

    def test_exact_boundary_and_inventory(self) -> None:
        doc = self.document
        self.assertEqual(doc.start, 0)
        self.assertEqual(doc.end, 490_077)
        self.assertEqual(doc.flags, 7)
        self.assertEqual(doc.song_size, 175_735)
        self.assertEqual(doc.sample_size, 6_640)
        self.assertEqual(doc.song_bpm_fixed, 196 << 16)
        self.assertEqual(len(doc.operations), 16_478)
        self.assertEqual(len(doc.events), 95)
        self.assertEqual(len(doc.splines), 11)
        self.assertEqual(len(doc.classes), 90)

    def test_party_multiply_extrusion(self) -> None:
        multiply = [op for op in self.document.operations if op["class_id"] in (0x95, 0x12F)]
        self.assertEqual(len(multiply), 819)
        self.assertTrue(all(op["parameters"][17] == 0 for op in multiply))


if __name__ == "__main__":
    unittest.main()
