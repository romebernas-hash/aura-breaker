"""Ki charge + ultimate-energy fix, as a thin subclass of ``battle_system.Battle``.

Why the ultimate felt impossible:
  * it cost 90 of 100 energy, but Transform (30) and Rage (25) are paid from the same pool,
  * while transformed there was NO energy regeneration at all (and a 3/s drain),
  * so after transforming the bar could never get back to 90.

What this module changes:
  * ULTIMATE_COST 90 -> 60  (transform then ultimate is possible straight away),
  * KI CHARGE: hold the power-up input to refill energy quickly (risky: you cannot act and the
    enemy keeps attacking, and the shield is unavailable while you power up),
  * a small regeneration while transformed so the drain is no longer a one-way trip,
  * an ``ultimate_ready`` event so the game can tell you the moment the ultimate is usable.

All numbers are here, in one place, for tuning.
"""
from __future__ import annotations

import config as C
from battle_system import Battle, EnemyPhase

ULTIMATE_COST = 60.0          # was 90
KI_CHARGE_RATE = 38.0         # energy / second while powering up (0 -> 60 in ~1.6 s)
TRANSFORMED_REGEN = 4.0       # energy / second while transformed and idle (drain is 3/s)

C.ULTIMATE_COST = ULTIMATE_COST   # battle_system reads C.ULTIMATE_COST at call time


class KiBattle(Battle):
    def _reset_round(self) -> None:
        super()._reset_round()
        self.ki_charging = False
        self.ki_request = False
        self._ult_was_ready = True

    # -- ki charge ---------------------------------------------------------
    def _can_ki_charge(self) -> bool:
        return (self.fsm.in_combat and self.fsm.is_idle and self.enemy_phase != EnemyPhase.DEFEATED
                and not self.shield_active and self.energy < C.PLAYER_MAX_ENERGY - 0.01)

    def _eval_ki(self) -> None:
        want = bool(self.ki_request) and self._can_ki_charge()
        if want and not self.ki_charging:
            self.ki_charging = True
            self.emit("ki_charge_start")
        elif not want and self.ki_charging:
            self.ki_charging = False
            self.emit("ki_charge_stop")

    def apply_commands(self, cmds, brow_mode: str = "auto") -> None:
        super().apply_commands(cmds, brow_mode)
        self._eval_ki()          # after the abilities, so starting a cast always wins over powering up

    # -- energy ------------------------------------------------------------
    def _update_energy(self, dt: float) -> None:
        super()._update_energy(dt)
        if self.ki_charging:
            self.energy = min(C.PLAYER_MAX_ENERGY, self.energy + KI_CHARGE_RATE * dt)
        elif self.transformed and self.fsm.is_idle:
            self.energy = min(C.PLAYER_MAX_ENERGY, self.energy + TRANSFORMED_REGEN * dt)

    def update(self, dt: float) -> None:
        super().update(dt)
        if not self.fsm.in_combat:
            if self.ki_charging:
                self.ki_charging = False
                self.emit("ki_charge_stop")
            return
        if self.ki_charging and not self._can_ki_charge():
            self._eval_ki()
        ready = self.cooldowns[self._ult] <= 0 and self.energy >= C.ULTIMATE_COST
        if ready and not self._ult_was_ready:
            self.emit("ultimate_ready")
        self._ult_was_ready = ready

    @property
    def _ult(self):
        from battle_system import Ability
        return Ability.ULTIMATE