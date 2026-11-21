"""Tests for KiBattle (ki charge + ultimate energy fix). Uses your REAL battle_system / game_state.

Run:  python -m unittest tests.test_battle_ki -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402
from battle_ki import KI_CHARGE_RATE, KiBattle  # noqa: E402
from battle_system import Ability  # noqa: E402


def fresh(**kw):
    b = KiBattle(training=False, **kw)
    b.drain_events()
    return b


class TestUltimateEnergy(unittest.TestCase):
    def test_cost_lowered_so_transform_then_ultimate_is_possible(self):
        self.assertLessEqual(C.ULTIMATE_COST + C.TRANSFORM_COST, C.PLAYER_MAX_ENERGY)

    def test_energy_recovers_while_transformed(self):
        b = fresh()
        b.transformed, b.transform_time, b.energy = True, 20.0, 40.0
        for _ in range(60 * 5):
            b.update(1 / 60)
        self.assertGreaterEqual(b.energy, 40.0)           # no longer a one-way drain


class TestKiCharge(unittest.TestCase):
    def test_holding_ki_charge_refills_energy_and_emits_events(self):
        b = fresh()
        b.energy = 10.0
        b.ki_request = True
        b._eval_ki()
        self.assertTrue(b.ki_charging)
        for _ in range(60):
            b.update(1 / 60)
        self.assertGreater(b.energy, 10 + KI_CHARGE_RATE * 0.8)
        kinds = [e.kind for e in b.drain_events()]
        self.assertIn("ki_charge_start", kinds)

    def test_releasing_stops_and_energy_is_capped(self):
        b = fresh()
        b.energy = 95.0
        b.ki_request = True
        b._eval_ki()
        for _ in range(120):
            b.update(1 / 60)
        self.assertLessEqual(b.energy, C.PLAYER_MAX_ENERGY)
        self.assertFalse(b.ki_charging)                   # full bar stops the power-up

    def test_cannot_ki_charge_while_casting(self):
        b = fresh()
        b.energy = 50.0
        b.request_transform()
        b.ki_request = True
        b._eval_ki()
        self.assertFalse(b.ki_charging)

    def test_ultimate_ready_event_after_recharging(self):
        b = fresh()
        b.energy = 20.0
        b.update(1 / 60)                                  # not ready
        b.drain_events()
        b.ki_request = True
        b._eval_ki()
        for _ in range(60 * 3):
            b.update(1 / 60)
        self.assertIn("ultimate_ready", [e.kind for e in b.drain_events()])
        self.assertTrue(b.can_use(Ability.ULTIMATE)[0])

    def test_restart_resets_ki_state(self):
        b = fresh()
        b.energy = 10.0
        b.ki_request = True
        b._eval_ki()
        b.restart()
        self.assertFalse(b.ki_charging)


if __name__ == "__main__":
    unittest.main(verbosity=2)