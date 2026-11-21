"""Geometry tests for the side-view fighter (needs pygame + config, like the game itself).

Run:  python -m unittest tests.test_fighter -v
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from fighter import FighterRenderer, ik2  # noqa: E402


class St:
    def __init__(self, name): self.name = name


class Fsm:
    def __init__(self, name): self.state = St(name)


class FakeBattle:
    def __init__(self, state="BATTLE_IDLE", **kw):
        self.fsm = Fsm(state)
        self.aim, self.charge, self.ult_time, self.rage_time, self.ki_charging = 0.0, 0.7, 0.0, 0.0, False
        self.__dict__.update(kw)


class Anchor:
    head = (320.0, 248.0)
    face_h = 170.0
    face_w = 130.0


ENEMY = (1010.0, 380.0)


def settle(state="BATTLE_IDLE", frames=45, **kw):
    f, b = FighterRenderer(), FakeBattle(state, **kw)
    for _ in range(frames):
        f.update(1 / 60, b, Anchor, ENEMY, None, 0.0)
    return f


class TestIK(unittest.TestCase):
    def test_bone_lengths_preserved(self):
        knee, end = ik2((0, 0), (120, 90), 100, 100, (1, 0))
        self.assertAlmostEqual(math.dist((0, 0), knee), 100, places=3)
        self.assertAlmostEqual(math.dist(knee, end), 100, places=3)

    def test_unreachable_target_is_clamped_not_nan(self):
        knee, end = ik2((0, 0), (900, 0), 70, 60, (0, 1))
        self.assertTrue(all(math.isfinite(v) for v in (*knee, *end)))
        self.assertLessEqual(math.dist((0, 0), end), 130.0)


class TestFacing(unittest.TestCase):
    def test_fighter_faces_the_enemy_in_every_pose(self):
        for state, extra in (("BATTLE_IDLE", {}), ("CHARGING", {}), ("ATTACKING", {}), ("DEFENDING", {}),
                             ("TRANSFORMING", {}), ("RAGE_BURST", {}), ("ULTIMATE", {"ult_time": 0.5}),
                             ("ULTIMATE", {"ult_time": 2.0}), ("VICTORY", {})):
            f = settle(state, **extra)
            j = f.joints
            self.assertGreater(j["ff"][0], j["fb"][0], state)           # front foot is toward the enemy
            self.assertGreater(j["hf"][0], j["pelvis"][0] - 60, state)  # front hand never trails far behind

    def test_ki_charge_pose_is_selected(self):
        self.assertEqual(settle("BATTLE_IDLE", ki_charging=True).pose, "ki")

    def test_all_joint_values_finite(self):
        f = settle("ATTACKING", aim=1.0)
        for v in f.joints.values():
            if isinstance(v, tuple):
                self.assertTrue(all(math.isfinite(c) for c in v))


class TestFiringHand(unittest.TestCase):
    def test_fire_snaps_arm_toward_the_enemy_immediately(self):
        f, b = FighterRenderer(), FakeBattle("BATTLE_IDLE")
        for _ in range(30):
            f.update(1 / 60, b, Anchor, ENEMY, None, 0.0)
        idle_hand = f.hand
        b.fsm = Fsm("ATTACKING")
        f.update(1 / 60, b, Anchor, ENEMY, None, 0.0)      # ONE frame later: beams spawn from this hand
        self.assertGreater(f.hand[0], idle_hand[0] + 40)
        sh = f.joints["sh"]
        ang = math.atan2(f.hand[1] - sh[1], f.hand[0] - sh[0])
        want = math.atan2(ENEMY[1] - sh[1], ENEMY[0] - sh[0])
        self.assertLess(abs(ang - want), math.radians(C.MAX_AIM_DEG) + 0.1)

    def test_aim_raises_and_lowers_the_arm(self):
        up = settle("ATTACKING", aim=1.0).hand[1]
        down = settle("ATTACKING", aim=-1.0).hand[1]
        self.assertLess(up, down)                           # +aim = higher on screen (smaller y)

    def test_ultimate_hands_move_from_hip_to_forward(self):
        charge = settle("ULTIMATE", ult_time=0.5).hand
        fire = settle("ULTIMATE", ult_time=2.0).hand
        self.assertGreater(fire[0], charge[0] + 120)
        self.assertLess(charge[1], 600)                     # cupped hands are above the ground


if __name__ == "__main__":
    unittest.main(verbosity=2)