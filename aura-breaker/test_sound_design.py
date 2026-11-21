"""Tests for the original sound pack (numpy only) and its wiring into the audio manager.

Run:  python -m unittest tests.test_sound_design -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sound_design as sd  # noqa: E402


class TestPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = {name: sd.render(name) for name in sd.GENERATORS}

    def test_every_sound_is_valid(self):
        for name, x in self.audio.items():
            self.assertTrue(np.isfinite(x).all(), name)
            self.assertLessEqual(float(np.max(np.abs(x))), 1.0, name)
            self.assertGreater(float(np.sqrt(np.mean(x ** 2))), 0.02, f"{name} is silent")
            self.assertGreater(len(x) / sd.SR, 0.05, name)

    def test_durations_match_the_game(self):
        d = lambda n: len(self.audio[n]) / sd.SR  # noqa: E731
        self.assertGreater(d("transform.wav"), 2.6)          # covers the 2.6 s transform animation + burst
        self.assertAlmostEqual(d("ultimate.wav"), 1.85, delta=0.1)   # peaks when ULTIMATE_CHARGE_TIME (1.8 s) ends
        self.assertGreater(d("ultimate_beam.wav"), 2.0)
        self.assertGreater(d("music.wav"), 20)

    def test_loops_have_no_click_at_the_seam(self):
        for name in sd.LOOP_FILES:
            x = self.audio[name]
            step = float(np.mean(np.abs(np.diff(x[:2000])))) or 1e-9
            self.assertLess(abs(x[0] - x[-1]) / step, 3.0, name)

    def test_loudness_balance(self):
        peak = lambda n: float(np.max(np.abs(self.audio[n])))  # noqa: E731
        self.assertLess(peak("ui.wav"), peak("impact.wav") * 0.5)
        self.assertLess(peak("deny.wav"), peak("beam.wav") * 0.5)
        self.assertLess(peak("music.wav"), peak("ultimate_beam.wav"))

    def test_charge_start_rises_in_loudness(self):
        x = np.abs(self.audio["charge_start.wav"])
        third = len(x) // 3
        self.assertGreater(x[third:2 * third].mean(), x[:third].mean())

    def test_transform_has_a_burst_at_the_climax(self):
        x = np.abs(self.audio["transform.wav"])
        a, b = int(2.0 * sd.SR), int(2.4 * sd.SR)
        c, e = int(2.6 * sd.SR), int(3.0 * sd.SR)
        self.assertGreater(x[c:e].max(), x[a:b].max() * 0.9)


class TestWriting(unittest.TestCase):
    def test_ensure_pack_generates_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertTrue(sd.ensure_sound_pack(d))
            self.assertTrue(all((d / n).exists() for n in sd.GENERATORS))
            self.assertFalse(sd.ensure_sound_pack(d))            # already current -> no regeneration
            (d / "beam.wav").unlink()
            self.assertTrue(sd.ensure_sound_pack(d))             # missing file -> regenerated

    def test_wav_format(self):
        import wave
        with tempfile.TemporaryDirectory() as tmp:
            sd.write_wav(Path(tmp) / "x.wav", sd.render("ui.wav"))
            with wave.open(str(Path(tmp) / "x.wav")) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 22050))


class TestAudioManagerWiring(unittest.TestCase):
    def test_every_mapped_file_has_a_generator_and_valid_category(self):
        try:
            import audio_manager as am
            import config as C
        except ImportError:
            self.skipTest("pygame / game modules not importable here")
        cats = set(getattr(C, "VOLUME_CHANNELS", ["music", "transform", "charge", "beam", "impact", "shield", "result"]))
        for name, (cat, fname) in am.SOUND_MAP.items():
            self.assertIn(fname, sd.GENERATORS, name)
            self.assertIn(cat, cats, name)
        self.assertEqual(set(f for _, f in am.SOUND_MAP.values()), set(sd.GENERATORS))


if __name__ == "__main__":
    unittest.main(verbosity=2)