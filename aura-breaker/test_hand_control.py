"""Tests for hand_controls (pure numpy; no camera, MediaPipe or pygame needed).

Run:  python -m unittest tests.test_hand_controls -v      (from the project root)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_controls import HandConfig, HandFrame, HandGestureEngine, HandObs, HandPose, classify_pose  # noqa: E402

DT = 1 / 30
SPREAD = (-0.45, -0.15, 0.15, 0.45)   # finger direction angles (radians from "up")


def make_hand(cx=0.5, cy=0.6, state="open", size=0.10, score=0.95) -> HandObs:
    """Synthetic 21-point hand. wrist at (cx, cy); fingers point up (negative y)."""
    lm = np.zeros((21, 3), np.float32)
    w = np.array([cx, cy])
    lm[0, :2] = w
    ratios = {"open": 2.0, "fist": 0.9, "point": None}[state] if state != "point" else None
    for i, ang in enumerate(SPREAD):
        d = np.array([np.sin(ang), -np.cos(ang)])
        mcp = w + d * size
        if state == "point":
            r = 2.0 if i == 0 else 0.9
        else:
            r = ratios
        tip = w + d * size * r
        pip = w + d * size * (1 + (r - 1) * 0.4)
        lm[5 + 4 * i, :2] = mcp
        lm[6 + 4 * i, :2] = pip
        lm[8 + 4 * i, :2] = tip
    lm[9, :2] = w + np.array([0.0, -size])   # middle MCP used for hand size
    lm[1:5, :2] = w + np.array([0.04, -0.03])  # thumb (ignored by the classifier)
    return HandObs(lm, score)


def run(engine, hands, seconds, dt=DT):
    frames = []
    for _ in range(int(round(seconds / dt))):
        frames.append(engine.update(hands() if callable(hands) else hands, dt))
    return frames


def eng():
    return HandGestureEngine(HandConfig(), aspect=1.0)


class TestClassifier(unittest.TestCase):
    def test_open_fist_other(self):
        self.assertEqual(classify_pose(make_hand(state="open").lm, aspect=1.0)[0], HandPose.OPEN)
        self.assertEqual(classify_pose(make_hand(state="fist").lm, aspect=1.0)[0], HandPose.FIST)
        self.assertEqual(classify_pose(make_hand(state="point").lm, aspect=1.0)[0], HandPose.OTHER)

    def test_scale_invariant(self):
        for size in (0.04, 0.10, 0.25):
            self.assertEqual(classify_pose(make_hand(state="open", size=size).lm, aspect=1.0)[0], HandPose.OPEN)
            self.assertEqual(classify_pose(make_hand(state="fist", size=size).lm, aspect=1.0)[0], HandPose.FIST)


class TestKiBlast(unittest.TestCase):
    def test_open_palm_charges_after_delay_and_fist_fires(self):
        e = eng()
        fr = run(e, [make_hand(state="open")], 0.6)
        self.assertTrue(fr[-1].charge_held)
        self.assertFalse(fr[0].charge_held)             # not instantly
        fr = run(e, [make_hand(state="fist")], 0.5)
        self.assertFalse(fr[-1].charge_held)            # release -> battle fires
        self.assertFalse(any(f.cancel_charge for f in fr))

    def test_fire_fist_never_raises_shield_until_released(self):
        e = eng()
        run(e, [make_hand(state="open")], 0.6)
        fr = run(e, [make_hand(state="fist")], 2.0)
        self.assertFalse(any(f.shield_held for f in fr))
        run(e, [make_hand(state="open")], 0.3)          # open again
        run(e, [make_hand(state="fist")], 0.1)
        # a fist right after open is again a fire fist; shield needs a fresh fist after a pause
        run(e, [make_hand(state="point")], 1.0)
        fr = run(e, [make_hand(state="fist")], 0.6)
        self.assertTrue(fr[-1].shield_held)

    def test_pose_flicker_does_not_release(self):
        e = eng()
        run(e, [make_hand(state="open")], 0.6)
        seq = [make_hand(state="open")] * 5 + [make_hand(state="fist")] + [make_hand(state="open")] * 5
        frames = [e.update([h], DT) for h in seq]
        self.assertTrue(all(f.charge_held for f in frames))

    def test_tracking_loss_short_keeps_long_cancels_without_firing(self):
        e = eng()
        run(e, [make_hand(state="open")], 0.6)
        fr = run(e, [], 0.3)
        self.assertTrue(all(f.charge_held for f in fr))
        self.assertFalse(any(f.cancel_charge for f in fr))
        run(e, [make_hand(state="open")], 0.2)           # recovers
        fr = run(e, [], 0.8)
        cancels = [f for f in fr if f.cancel_charge]
        self.assertEqual(len(cancels), 1)                # exactly one cancel event
        self.assertFalse(fr[-1].charge_held)

    def test_low_confidence_hand_is_ignored(self):
        e = eng()
        fr = run(e, [make_hand(state="open", score=0.2)], 0.6)
        self.assertFalse(any(f.charge_held for f in fr))
        self.assertEqual(fr[-1].hands, 0)


class TestShield(unittest.TestCase):
    def test_fist_from_scratch_raises_shield_after_hold(self):
        e = eng()
        fr = run(e, [make_hand(state="fist")], 0.2)
        self.assertFalse(fr[-1].shield_held)
        fr = run(e, [make_hand(state="fist")], 0.5)
        self.assertTrue(fr[-1].shield_held)
        self.assertFalse(fr[-1].charge_held)

    def test_shield_drops_when_fist_opens(self):
        e = eng()
        run(e, [make_hand(state="fist")], 0.8)
        fr = run(e, [make_hand(state="point")], 0.5)
        self.assertFalse(fr[-1].shield_held)

    def test_brief_closing_does_not_trigger_shield(self):
        e = eng()
        fr = run(e, [make_hand(state="fist")], 0.25)
        fr += run(e, [make_hand(state="point")], 0.5)
        self.assertFalse(any(f.shield_held for f in fr))


class TestUltimate(unittest.TestCase):
    def pair(self, gap=0.12, state="open"):
        return [make_hand(cx=0.5 - gap / 2, state=state), make_hand(cx=0.5 + gap / 2, state=state)]

    def test_two_hands_together_fires_once_per_hold(self):
        e = eng()
        fr = run(e, self.pair(), 3.0)
        self.assertEqual(sum(f.ultimate for f in fr), 1)     # edge only, never repeats while held
        self.assertFalse(any(f.charge_held or f.shield_held for f in fr))

    def test_needs_hold_time(self):
        e = eng()
        fr = run(e, self.pair(), 0.3)
        self.assertFalse(any(f.ultimate for f in fr))
        self.assertTrue(0.0 < fr[-1].ultimate_progress < 1.0)

    def test_hands_far_apart_no_ultimate(self):
        e = eng()
        fr = run(e, self.pair(gap=0.7), 2.0)
        self.assertFalse(any(f.ultimate for f in fr))

    def test_rearm_requires_separation_and_cooldown(self):
        e = eng()
        run(e, self.pair(), 1.0)
        run(e, self.pair(gap=0.7), 0.6)                       # apart (> loss grace)
        fr = run(e, self.pair(), 1.2)
        self.assertEqual(sum(f.ultimate for f in fr), 0)      # still within cooldown window
        fr = run(e, self.pair(gap=0.7), 2.0)
        fr = run(e, self.pair(), 1.2)
        self.assertEqual(sum(f.ultimate for f in fr), 1)

    def test_two_fists_together_is_shield_not_ultimate(self):
        e = eng()
        fr = run(e, self.pair(state="fist"), 1.5)
        self.assertFalse(any(f.ultimate for f in fr))
        self.assertTrue(fr[-1].shield_held)

    def test_second_hand_arriving_during_charge_cancels_instead_of_firing(self):
        e = eng()
        run(e, [make_hand(cx=0.3, state="open")], 0.6)
        frames = run(e, [make_hand(cx=0.46, state="open"), make_hand(cx=0.54, state="open")], 0.3)
        self.assertTrue(any(f.cancel_charge for f in frames))
        self.assertFalse(frames[-1].charge_held)

    def test_hand_lost_during_ultimate_pose_resets_progress(self):
        e = eng()
        run(e, self.pair(), 0.4)
        run(e, [], 0.5)
        fr = run(e, self.pair(), 0.3)
        self.assertFalse(any(f.ultimate for f in fr))


class TestAimPoint(unittest.TestCase):
    def test_aim_point_tracks_hand_and_midpoint(self):
        e = eng()
        f = run(e, [make_hand(cx=0.2, cy=0.5)], 0.2)[-1]
        self.assertIsNotNone(f.aim_point)
        self.assertLess(f.aim_point[0], 0.3)
        f = run(e, [make_hand(cx=0.45), make_hand(cx=0.55)], 0.2)[-1]
        self.assertAlmostEqual(f.aim_point[0], 0.5, delta=0.03)


class TestEngineRobustness(unittest.TestCase):
    def test_reset_clears_everything(self):
        e = eng()
        run(e, [make_hand(state="open")], 0.6)
        e.reset()
        f = e.update([], DT)
        self.assertFalse(f.charge_held or f.shield_held or f.ultimate)

    def test_garbage_input_does_not_crash(self):
        e = eng()
        bad = HandObs(np.zeros((5, 3), np.float32), 0.9)
        f = e.update([bad], DT)
        self.assertEqual(f.hands, 0)
        f = e.update(None, DT)
        self.assertIsInstance(f, HandFrame)

    def test_huge_dt_is_clamped(self):
        e = eng()
        f = e.update([make_hand(state="open")], 5.0)
        self.assertFalse(f.charge_held)    # a hitch must not skip the debounce


if __name__ == "__main__":
    unittest.main(verbosity=2)