"""Battle logic: abilities, energy, damage, shield mitigation, enemy AI.

This module is pure game logic (no pygame, no camera) so it can be unit tested
headlessly. Visual/audio systems react to the :class:`Event` objects it emits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import config as C
from game_state import Commands, GameState as S, StateMachine


class Ability(IntEnum):
    TRANSFORM = 1
    BLASTER = 2
    SCANNER = 3
    RAGE = 4
    SHIELD = 5
    ULTIMATE = 6


ABILITY_LABEL = {
    Ability.TRANSFORM: "TRANSFORM",
    Ability.BLASTER: "BLASTER",
    Ability.SCANNER: "SCANNER",
    Ability.RAGE: "RAGE",
    Ability.SHIELD: "SHIELD",
    Ability.ULTIMATE: "ULTIMATE",
}

COOLDOWN_TOTAL = {
    Ability.TRANSFORM: C.TRANSFORM_COOLDOWN,
    Ability.BLASTER: C.BLASTER_COOLDOWN,
    Ability.SCANNER: C.SCAN_COOLDOWN + C.SCAN_DURATION,
    Ability.RAGE: C.RAGE_COOLDOWN,
    Ability.SHIELD: C.SHIELD_BREAK_LOCK,
    Ability.ULTIMATE: C.ULTIMATE_COOLDOWN,
}


class EnemyPhase(Enum):
    IDLE = "idle"
    TELEGRAPH = "telegraph"
    ATTACK = "attack"
    RECOVER = "recover"
    DEFEATED = "defeated"


@dataclass
class Event:
    kind: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EnemyStats:
    hp: float
    interval: float
    telegraph: float
    damage: float
    power: int


# --------------------------------------------------------------------------
# Pure helper functions (unit tested)
# --------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def blaster_damage(charge: float, multiplier: float = 1.0) -> float:
    c = clamp(charge, 0.0, 1.0)
    return (C.BLASTER_DMG_MIN + (C.BLASTER_DMG_MAX - C.BLASTER_DMG_MIN) * c) * max(0.0, multiplier)


def blaster_cost(charge: float) -> float:
    c = clamp(charge, 0.0, 1.0)
    return C.BLASTER_COST_MIN + (C.BLASTER_COST_MAX - C.BLASTER_COST_MIN) * c


def apply_shield(raw: float, active: bool, strength: float) -> tuple[float, float]:
    """Return ``(damage_taken, new_strength)`` for an incoming hit."""
    raw = max(0.0, raw)
    if not active or strength <= 0:
        return raw, max(0.0, strength)
    mitigated = raw * C.SHIELD_REDUCTION
    absorbed = min(mitigated, strength)
    return raw - absorbed, strength - absorbed


def power_level(energy: float, transform_mult: float = 1.0, upgrades: int = 0, rage_mult: float = 1.0) -> int:
    frac = clamp(energy / C.PLAYER_MAX_ENERGY, 0.0, 1.0)
    value = C.BASE_POWER * (0.6 + 0.4 * frac) * transform_mult * rage_mult * (1 + C.UPGRADE_POWER_BONUS * max(0, upgrades))
    return int(value)


def combo_multiplier(combo: int) -> float:
    return min(C.COMBO_MAX_MULT, 1.0 + C.COMBO_STEP * max(0, combo))


def enemy_stats(round_number: int) -> EnemyStats:
    n = max(1, int(round_number)) - 1
    return EnemyStats(
        hp=C.ENEMY_BASE_HP * C.ENEMY_HP_SCALE ** n,
        interval=max(C.ENEMY_MIN_INTERVAL, C.ENEMY_BASE_INTERVAL * C.ENEMY_INTERVAL_SCALE ** n),
        telegraph=max(C.ENEMY_MIN_TELEGRAPH, C.ENEMY_BASE_TELEGRAPH * C.ENEMY_TELEGRAPH_SCALE ** n),
        damage=C.ENEMY_BASE_DAMAGE * C.ENEMY_DAMAGE_SCALE ** n,
        power=int(C.ENEMY_BASE_POWER * (1 + 0.15 * n)),
    )


# --------------------------------------------------------------------------
class Battle:
    """One battle session (survives across rounds until restart)."""

    def __init__(
        self,
        fsm: StateMachine | None = None,
        training: bool = False,
        round_number: int = 1,
        rng: random.Random | None = None,
    ) -> None:
        self.training = training
        self.idle_state = S.TRAINING if training else S.BATTLE_IDLE
        self.fsm = fsm if fsm is not None else StateMachine(self.idle_state)
        self.rng = rng or random.Random()
        self.events: list[Event] = []
        self.round = max(1, round_number)
        self.upgrades = self.round - 1
        self.score = 0
        self.player_hp = C.PLAYER_MAX_HP
        self.energy = C.PLAYER_MAX_ENERGY
        self._reset_round()

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------
    def _reset_round(self) -> None:
        st = enemy_stats(self.round)
        self.stats = st
        self.enemy_max_hp = st.hp
        self.enemy_hp = st.hp
        self.enemy_phase = EnemyPhase.IDLE
        self.enemy_timer = st.interval + 1.0
        self.enemy_hit_timer = 0.0
        self.cooldowns: dict[Ability, float] = {a: 0.0 for a in Ability}
        self.transformed = False
        self.transform_time = 0.0
        self.rage_time = 0.0
        self.scan_time = 0.0
        self.anim_time = 0.0
        self.attack_time = 0.0
        self.ult_time = 0.0
        self._ult_fired = False
        self._ult_landed = False
        self.shield_active = False
        self.shield_strength = C.SHIELD_MAX_STRENGTH
        self.shield_lock = 0.0
        self.charge = 0.0
        self.aim = 0.0
        self.combo = 0
        self.combo_timer = 0.0
        self._scheduled: list[list] = []
        self._charge_latched = False
        self._charge_prev = False
        self._shield_prev = False

    def restart(self) -> None:
        """Full restart: round 1, score 0, full health."""
        self.round = 1
        self.upgrades = 0
        self.score = 0
        self.player_hp = C.PLAYER_MAX_HP
        self.energy = C.PLAYER_MAX_ENERGY
        self._reset_round()
        self.events.clear()
        self.fsm.force(self.idle_state)

    def next_round(self) -> None:
        """Advance after a victory: harder enemy, small heal, keep score."""
        self.round += 1
        self.upgrades += 1
        self.player_hp = min(C.PLAYER_MAX_HP, self.player_hp + C.ROUND_HEAL)
        self.energy = C.PLAYER_MAX_ENERGY
        score = self.score
        self._reset_round()
        self.score = score
        if not self.fsm.transition(self.idle_state):
            self.fsm.force(self.idle_state)

    # ------------------------------------------------------------------
    # Events / helpers
    # ------------------------------------------------------------------
    def emit(self, kind: str, **data) -> None:
        self.events.append(Event(kind, data))

    def drain_events(self) -> list[Event]:
        ev, self.events = self.events, []
        return ev

    def schedule(self, delay: float, kind: str, **data) -> None:
        self._scheduled.append([delay, kind, data])

    @property
    def damage_multiplier(self) -> float:
        return (C.TRANSFORM_MULT if self.transformed else 1.0) * (C.RAGE_MULT if self.rage_time > 0 else 1.0)

    @property
    def power(self) -> int:
        return power_level(
            self.energy,
            C.TRANSFORM_MULT if self.transformed else 1.0,
            self.upgrades,
            C.RAGE_MULT if self.rage_time > 0 else 1.0,
        )

    @property
    def enemy_power(self) -> int:
        frac = self.enemy_hp / self.enemy_max_hp if self.enemy_max_hp else 0
        return int(self.stats.power * (0.5 + 0.5 * frac))

    @property
    def invulnerable(self) -> bool:
        return self.fsm.state.name in C.INVULN_STATES

    def cooldown_fraction(self, ability: Ability) -> float:
        total = COOLDOWN_TOTAL[ability]
        return clamp(self.cooldowns[ability] / total, 0.0, 1.0) if total > 0 else 0.0

    def ability_ready(self, ability: Ability) -> bool:
        return self.can_use(ability)[0]

    def _deny(self, ability: Ability, reason: str, silent: bool = False) -> bool:
        if not silent:
            self.emit("denied", ability=ability, reason=reason, remaining=self.cooldowns.get(ability, 0.0))
        return False

    # ------------------------------------------------------------------
    # Ability validation
    # ------------------------------------------------------------------
    def can_use(self, ability: Ability) -> tuple[bool, str]:
        if not self.fsm.in_combat:
            return False, "not_in_battle"
        idle = self.fsm.is_idle
        cd = self.cooldowns[ability]
        if ability == Ability.SCANNER:
            if self.scan_time > 0:
                return False, "active"
            if self.energy < C.SCAN_COST:
                return False, "energy"
            return (False, "cooldown") if cd > 0 else (True, "")
        if not idle:
            return False, "busy"
        if ability == Ability.TRANSFORM:
            if self.transformed:
                return False, "active"
            if cd > 0:
                return False, "cooldown"
            return (True, "") if self.energy >= C.TRANSFORM_COST else (False, "energy")
        if ability == Ability.RAGE:
            if self.rage_time > 0:
                return False, "active"
            if cd > 0:
                return False, "cooldown"
            return (True, "") if self.energy >= C.RAGE_COST else (False, "energy")
        if ability == Ability.BLASTER:
            if cd > 0:
                return False, "cooldown"
            return (True, "") if self.energy >= C.BLASTER_MIN_ENERGY else (False, "energy")
        if ability == Ability.SHIELD:
            if self.shield_lock > 0:
                return False, "cooldown"
            if self.shield_strength < C.SHIELD_MIN_STRENGTH:
                return False, "shield_weak"
            return (True, "") if self.energy >= C.SHIELD_MIN_ENERGY else (False, "energy")
        if ability == Ability.ULTIMATE:
            if cd > 0:
                return False, "cooldown"
            return (True, "") if self.energy >= C.ULTIMATE_COST else (False, "energy")
        return False, "unknown"

    # ------------------------------------------------------------------
    # Ability requests
    # ------------------------------------------------------------------
    def request_transform(self) -> bool:
        ok, why = self.can_use(Ability.TRANSFORM)
        if not ok:
            return self._deny(Ability.TRANSFORM, why)
        if not self.fsm.transition(S.TRANSFORMING):
            return self._deny(Ability.TRANSFORM, "busy")
        self.energy -= C.TRANSFORM_COST
        self.anim_time = C.TRANSFORM_ANIM
        self.cooldowns[Ability.TRANSFORM] = C.TRANSFORM_COOLDOWN
        self.emit("transform_start", duration=C.TRANSFORM_ANIM)
        return True

    def request_rage(self) -> bool:
        ok, why = self.can_use(Ability.RAGE)
        if not ok:
            return self._deny(Ability.RAGE, why)
        if not self.fsm.transition(S.RAGE_BURST):
            return self._deny(Ability.RAGE, "busy")
        self.energy -= C.RAGE_COST
        self.anim_time = C.RAGE_ANIM
        self.cooldowns[Ability.RAGE] = C.RAGE_COOLDOWN
        self.emit("rage_start", duration=C.RAGE_ANIM)
        return True

    def brow_action(self, mode: str = "auto") -> bool:
        """Intense-brow gesture. Priority: transformation first, then rage."""
        if mode == "transform":
            return self.request_transform()
        if mode == "rage":
            return self.request_rage()
        if not self.transformed and self.can_use(Ability.TRANSFORM)[0]:
            return self.request_transform()
        if self.can_use(Ability.RAGE)[0]:
            return self.request_rage()
        # Neither available: report the more informative reason once.
        target = Ability.RAGE if self.transformed else Ability.TRANSFORM
        return self._deny(target, self.can_use(target)[1])

    def request_scan(self) -> bool:
        ok, why = self.can_use(Ability.SCANNER)
        if not ok:
            return self._deny(Ability.SCANNER, why)
        self.energy -= C.SCAN_COST
        self.scan_time = C.SCAN_DURATION
        self.cooldowns[Ability.SCANNER] = C.SCAN_COOLDOWN + C.SCAN_DURATION
        self.emit("scan_on")
        return True

    def request_ultimate(self) -> bool:
        ok, why = self.can_use(Ability.ULTIMATE)
        if not ok:
            return self._deny(Ability.ULTIMATE, why)
        if not self.fsm.transition(S.ULTIMATE):
            return self._deny(Ability.ULTIMATE, "busy")
        self.energy -= C.ULTIMATE_COST
        self.cooldowns[Ability.ULTIMATE] = C.ULTIMATE_COOLDOWN
        self.ult_time = 0.0
        self._ult_fired = False
        self._ult_landed = False
        self.emit("ultimate_start", duration=C.ULTIMATE_TOTAL_TIME)
        return True

    # -- blaster ---------------------------------------------------------
    def start_charge(self, silent: bool = False) -> bool:
        ok, why = self.can_use(Ability.BLASTER)
        if not ok:
            return self._deny(Ability.BLASTER, why, silent)
        if not self.fsm.transition(S.CHARGING):
            return self._deny(Ability.BLASTER, "busy", silent)
        self.charge = 0.0
        self.emit("charge_start")
        return True

    def _charge_cap(self) -> float:
        span = C.BLASTER_COST_MAX - C.BLASTER_COST_MIN
        return clamp((self.energy - C.BLASTER_COST_MIN) / span, 0.0, 1.0) if span > 0 else 1.0

    def release_charge(self) -> bool:
        if self.fsm.state != S.CHARGING:
            return False
        if self.charge < C.BLASTER_MIN_CHARGE:
            self.charge = 0.0
            self.fsm.transition(self.idle_state)
            self.emit("charge_cancel")
            return False
        return self._fire_blaster()

    def _fire_blaster(self) -> bool:
        c = clamp(self.charge, 0.0, 1.0)
        cost = blaster_cost(c)
        self.energy = max(0.0, self.energy - cost)
        dmg = blaster_damage(c, self.damage_multiplier)
        hit = abs(self.aim) <= C.AIM_TOLERANCE
        if not self.fsm.transition(S.ATTACKING):
            return False
        self.attack_time = C.BEAM_DURATION
        self.cooldowns[Ability.BLASTER] = C.BLASTER_COOLDOWN
        self.emit("beam_fire", charge=c, damage=dmg, aim=self.aim, hit=hit, cost=cost)
        self.schedule(C.BEAM_TRAVEL_TIME, "beam_land", damage=dmg, hit=hit, charge=c)
        self.charge = 0.0
        return True

    # -- shield ----------------------------------------------------------
    def _shield_on(self, silent: bool) -> bool:
        ok, why = self.can_use(Ability.SHIELD)
        if not ok:
            return self._deny(Ability.SHIELD, why, silent)
        if not self.fsm.transition(S.DEFENDING):
            return False
        self.shield_active = True
        self.emit("shield_on")
        return True

    def _shield_off(self, broken: bool = False) -> None:
        if not self.shield_active:
            return
        self.shield_active = False
        if broken:
            self.shield_strength = 0.0
            self.shield_lock = C.SHIELD_BREAK_LOCK
            self.cooldowns[Ability.SHIELD] = C.SHIELD_BREAK_LOCK
            self.emit("shield_break")
        else:
            self.emit("shield_off")
        if self.fsm.state == S.DEFENDING:
            self.fsm.transition(self.idle_state)

    def set_shield(self, held: bool) -> None:
        rising = held and not self._shield_prev
        self._shield_prev = held
        if held and not self.shield_active:
            if self.fsm.is_idle:
                self._shield_on(silent=not rising)
        elif not held and self.shield_active:
            self._shield_off()

    # -- input -----------------------------------------------------------
    def set_aim(self, aim: float) -> None:
        self.aim = clamp(aim, -1.0, 1.0)

    def _handle_charge_input(self, held: bool) -> None:
        rising = held and not self._charge_prev
        self._charge_prev = held
        if held:
            if self.fsm.is_idle and not self._charge_latched:
                self.start_charge(silent=not rising)
        else:
            self._charge_latched = False
            if self.fsm.state == S.CHARGING:
                self.release_charge()

    def apply_commands(self, cmds: Commands, brow_mode: str = "auto") -> None:
        """Feed one frame of input. Edge commands should be True for one frame."""
        if not self.fsm.in_combat:
            return
        if cmds.ultimate:
            self.request_ultimate()
        if cmds.transform:
            self.request_transform()
        if cmds.rage:
            self.request_rage()
        if cmds.brow_action:
            self.brow_action(brow_mode)
        if cmds.scan:
            self.request_scan()
        self.set_aim(cmds.aim)
        self._handle_charge_input(cmds.charge_held)
        self.set_shield(cmds.shield_held)

    # ------------------------------------------------------------------
    # Damage
    # ------------------------------------------------------------------
    def damage_enemy(self, dmg: float, source: str) -> None:
        if self.enemy_phase == EnemyPhase.DEFEATED or dmg <= 0:
            return
        self.enemy_hp = max(0.0, self.enemy_hp - dmg)
        self.combo += 1
        self.combo_timer = C.COMBO_WINDOW
        self.score += int(dmg * C.SCORE_PER_DAMAGE * combo_multiplier(self.combo))
        self.enemy_hit_timer = C.ENEMY_HIT_TIME
        self.emit("enemy_hit", damage=dmg, source=source, hp=self.enemy_hp, combo=self.combo)
        if self.enemy_phase == EnemyPhase.TELEGRAPH and dmg >= C.STAGGER_DAMAGE:
            self.enemy_phase = EnemyPhase.RECOVER
            self.enemy_timer = C.ENEMY_RECOVER_TIME * 1.5
            self.emit("enemy_stagger")

    def damage_player(self, raw: float) -> float:
        """Apply an enemy hit to the player. Returns damage actually taken."""
        if self.training:
            return 0.0
        if self.invulnerable:
            self.emit("player_dodged", damage=raw)
            return 0.0
        taken, new_strength = apply_shield(raw, self.shield_active, self.shield_strength)
        absorbed = raw - taken
        if self.shield_active and absorbed > 0:
            self.shield_strength = new_strength
            self.emit("shield_hit", absorbed=absorbed)
            if self.shield_strength <= 0:
                self._shield_off(broken=True)
        self.player_hp = max(0.0, self.player_hp - taken)
        if taken > 0:
            self.combo = 0
        self.emit("player_hit", damage=taken, raw=raw, shielded=absorbed > 0)
        return taken

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if not self.fsm.in_combat or dt <= 0:
            return
        dt = min(dt, C.MAX_DT * 2)
        for a in Ability:
            if self.cooldowns[a] > 0:
                self.cooldowns[a] = max(0.0, self.cooldowns[a] - dt)
        self.shield_lock = max(0.0, self.shield_lock - dt)
        self._update_scheduled(dt)
        self._update_buffs(dt)
        self._update_actions(dt)
        self._update_energy(dt)
        self._update_enemy(dt)
        if self.combo > 0:
            self.combo_timer -= dt
            if self.combo_timer <= 0:
                self.combo = 0
        self._check_end()

    def _update_scheduled(self, dt: float) -> None:
        due, rest = [], []
        for item in self._scheduled:
            item[0] -= dt
            (due if item[0] <= 0 else rest).append(item)
        self._scheduled = rest
        for _, kind, data in due:
            if kind == "beam_land":
                if data["hit"]:
                    self.damage_enemy(data["damage"], "blaster")
                else:
                    self.combo = 0
                    self.emit("beam_miss")
            elif kind == "enemy_projectile":
                self.damage_player(data["damage"])
            elif kind == "ultimate_land":
                self.damage_enemy(data["damage"], "ultimate")

    def _update_buffs(self, dt: float) -> None:
        if self.transformed:
            self.transform_time -= dt
            self.energy -= C.TRANSFORM_DRAIN * dt
            if self.transform_time <= 0 or self.energy <= 0:
                self.energy = max(0.0, self.energy)
                self.transformed = False
                self.transform_time = 0.0
                self.emit("transform_end")
        if self.rage_time > 0:
            self.rage_time = max(0.0, self.rage_time - dt)
            if self.rage_time == 0:
                self.emit("rage_end")
        if self.scan_time > 0:
            self.scan_time = max(0.0, self.scan_time - dt)
            if self.scan_time == 0:
                self.emit("scan_off")

    def _update_actions(self, dt: float) -> None:
        st = self.fsm.state
        if st == S.TRANSFORMING:
            self.anim_time -= dt
            if self.anim_time <= 0:
                self.transformed = True
                self.transform_time = C.TRANSFORM_DURATION
                self.emit("transform_complete", multiplier=C.TRANSFORM_MULT, power=self.power)
                self.fsm.transition(self.idle_state)
        elif st == S.RAGE_BURST:
            self.anim_time -= dt
            if self.anim_time <= 0:
                self.rage_time = C.RAGE_DURATION
                self.emit("rage_active", multiplier=C.RAGE_MULT)
                self.fsm.transition(self.idle_state)
        elif st == S.CHARGING:
            cap = self._charge_cap()
            self.charge = min(cap, self.charge + dt / C.BLASTER_CHARGE_TIME)
            if self.charge >= cap - 1e-9:
                if self.charge >= C.BLASTER_MIN_CHARGE:
                    self._charge_latched = True
                    self._fire_blaster()
                else:
                    self.release_charge()
        elif st == S.ATTACKING:
            self.attack_time -= dt
            if self.attack_time <= 0:
                self.fsm.transition(self.idle_state)
        elif st == S.DEFENDING:
            self.energy -= C.SHIELD_DRAIN * dt
            if self.energy <= 0:
                self.energy = 0.0
                self._shield_off(broken=False)
        elif st == S.ULTIMATE:
            self.ult_time += dt
            if not self._ult_fired and self.ult_time >= C.ULTIMATE_CHARGE_TIME:
                self._ult_fired = True
                dmg = C.ULTIMATE_DAMAGE * self.damage_multiplier
                self.emit("ultimate_fire", damage=dmg)
                self.schedule(0.30, "ultimate_land", damage=dmg)
            if self.ult_time >= C.ULTIMATE_TOTAL_TIME:
                self.emit("ultimate_end")
                self.fsm.transition(self.idle_state)

    def _update_energy(self, dt: float) -> None:
        if self.shield_active and self.fsm.state != S.DEFENDING:
            self.shield_active = False
        if not self.shield_active and self.shield_lock <= 0 and self.shield_strength < C.SHIELD_MAX_STRENGTH:
            self.shield_strength = min(C.SHIELD_MAX_STRENGTH, self.shield_strength + C.SHIELD_REGEN * dt)
        st = self.fsm.state
        if self.transformed or st in (S.DEFENDING, S.CHARGING, S.ULTIMATE):
            pass
        else:
            regen = C.TRAINING_ENERGY_REGEN if self.training else C.ENERGY_REGEN
            self.energy += regen * dt
        self.energy = clamp(self.energy, 0.0, C.PLAYER_MAX_ENERGY)

    def _update_enemy(self, dt: float) -> None:
        if self.enemy_phase == EnemyPhase.DEFEATED:
            return
        self.enemy_hit_timer = max(0.0, self.enemy_hit_timer - dt)
        if self.training:
            return
        self.enemy_timer -= dt
        if self.enemy_timer > 0:
            return
        ph = self.enemy_phase
        if ph == EnemyPhase.IDLE:
            self.enemy_phase = EnemyPhase.TELEGRAPH
            self.enemy_timer = self.stats.telegraph
            self.emit("enemy_telegraph", duration=self.stats.telegraph)
        elif ph == EnemyPhase.TELEGRAPH:
            self.enemy_phase = EnemyPhase.ATTACK
            self.enemy_timer = C.ENEMY_ATTACK_ANIM
            self.emit("enemy_attack", travel=C.ENEMY_PROJECTILE_TIME, damage=self.stats.damage)
            self.schedule(C.ENEMY_PROJECTILE_TIME, "enemy_projectile", damage=self.stats.damage)
        elif ph == EnemyPhase.ATTACK:
            self.enemy_phase = EnemyPhase.RECOVER
            self.enemy_timer = C.ENEMY_RECOVER_TIME
        elif ph == EnemyPhase.RECOVER:
            self.enemy_phase = EnemyPhase.IDLE
            self.enemy_timer = self.stats.interval * self.rng.uniform(0.85, 1.15)

    def _clear_actions(self) -> None:
        self.charge = 0.0
        self.shield_active = False
        self.scan_time = 0.0

    def _check_end(self) -> None:
        if self.enemy_hp <= 0:
            if self.training:
                self.enemy_hp = self.enemy_max_hp
                self.emit("dummy_reset")
                return
            self.enemy_phase = EnemyPhase.DEFEATED
            self._clear_actions()
            self.score += C.VICTORY_BONUS * self.round
            if self.fsm.transition(S.VICTORY):
                self.emit("victory", score=self.score, round=self.round)
            return
        if self.player_hp <= 0 and not self.training:
            self._clear_actions()
            if self.fsm.transition(S.DEFEAT):
                self.emit("defeat", score=self.score, round=self.round)
