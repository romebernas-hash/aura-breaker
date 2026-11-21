import random

import pytest

import config as C
from battle_system import (Ability, Battle, EnemyPhase, apply_shield, blaster_cost, blaster_damage, combo_multiplier,
                           enemy_stats, power_level)
from game_state import Commands, GameState as S, StateMachine

DT = 1 / 60


def new(training=False, **kw):
    b = Battle(training=training, rng=random.Random(1), **kw)
    b.enemy_timer = 9999     # keep the enemy passive unless a test wants it
    return b


def step(b, seconds, cmds=None):
    for _ in range(int(seconds / DT)):
        b.apply_commands(cmds or Commands())
        b.update(DT)


def kinds(b):
    return [e.kind for e in b.events]


# ------------------------------------------------------------------ pure maths
def test_blaster_damage_scales_with_charge_and_multiplier():
    assert blaster_damage(0) == C.BLASTER_DMG_MIN
    assert blaster_damage(1) == C.BLASTER_DMG_MAX
    assert blaster_damage(0.5, 2.0) == pytest.approx((C.BLASTER_DMG_MIN + C.BLASTER_DMG_MAX) / 2 * 2)


@pytest.mark.parametrize("c", [-5, 5, float("inf")])
def test_blaster_values_are_bounded(c):
    assert C.BLASTER_DMG_MIN <= blaster_damage(c) <= C.BLASTER_DMG_MAX
    assert C.BLASTER_COST_MIN <= blaster_cost(c) <= C.BLASTER_COST_MAX


def test_shield_reduction_and_depletion():
    taken, s = apply_shield(20, True, 60)
    assert taken == pytest.approx(20 * (1 - C.SHIELD_REDUCTION))
    assert s == pytest.approx(60 - 20 * C.SHIELD_REDUCTION)
    assert apply_shield(20, False, 60) == (20, 60)          # inactive: full damage
    assert apply_shield(20, True, 0) == (20, 0)             # broken: full damage
    taken, s = apply_shield(100, True, 10)                  # overflow passes through
    assert s == 0 and taken == pytest.approx(90)
    assert apply_shield(-5, True, 10)[0] == 0               # invalid damage


def test_power_level_and_combo_bounds():
    assert power_level(100) > power_level(0) > 0
    assert power_level(100, transform_mult=1.5) == pytest.approx(power_level(100) * 1.5, rel=0.001)
    assert power_level(1000) == power_level(100)            # energy clamped
    assert combo_multiplier(10_000) == C.COMBO_MAX_MULT and combo_multiplier(-3) == 1.0


def test_enemy_difficulty_scaling():
    a, b = enemy_stats(1), enemy_stats(5)
    assert b.hp > a.hp and b.damage > a.damage
    assert b.interval < a.interval and b.telegraph < a.telegraph
    assert enemy_stats(500).interval == C.ENEMY_MIN_INTERVAL and enemy_stats(500).telegraph == C.ENEMY_MIN_TELEGRAPH
    assert enemy_stats(0).hp == a.hp


# ------------------------------------------------------------------ transformation / rage
def test_transformation_flow_energy_and_multiplier():
    b = new()
    assert b.request_transform() and b.fsm.state == S.TRANSFORMING
    assert b.energy == C.PLAYER_MAX_ENERGY - C.TRANSFORM_COST
    assert not b.request_transform()                         # no re-activation during the animation
    assert "denied" in kinds(b)
    step(b, C.TRANSFORM_ANIM + 0.2)
    assert b.transformed and b.fsm.state == S.BATTLE_IDLE
    assert b.damage_multiplier == C.TRANSFORM_MULT
    assert "transform_complete" in kinds(b)


def test_transformation_expires_and_drains_energy():
    b = new()
    b.request_transform()
    step(b, C.TRANSFORM_ANIM + 1)
    e0 = b.energy
    step(b, 2.0)
    assert b.energy < e0                                     # sustained cost, no regen
    b.energy = 0.5
    step(b, 1.0)
    assert not b.transformed and "transform_end" in kinds(b)


def test_transform_requires_energy_and_respects_cooldown():
    b = new()
    b.energy = C.TRANSFORM_COST - 1
    assert not b.request_transform()
    b.energy = 100
    b.request_transform()
    step(b, C.TRANSFORM_ANIM + 0.1)
    b.transformed, b.transform_time = False, 0
    assert not b.request_transform()                         # still on cooldown
    assert b.can_use(Ability.TRANSFORM)[1] == "cooldown"


def test_brow_action_priority_transform_then_rage():
    b = new()
    assert b.brow_action("auto") and b.fsm.state == S.TRANSFORMING
    step(b, C.TRANSFORM_ANIM + 0.1)
    assert b.brow_action("auto") and b.fsm.state == S.RAGE_BURST    # second use -> rage
    # never both at once
    assert not b.request_transform() and not b.request_rage()


def test_rage_boosts_damage_and_stacks_with_transform():
    b = new()
    b.request_rage()
    step(b, C.RAGE_ANIM + 0.1)
    assert b.rage_time > 0 and b.damage_multiplier == C.RAGE_MULT
    b.transformed = True
    assert b.damage_multiplier == pytest.approx(C.RAGE_MULT * C.TRANSFORM_MULT)
    step(b, C.RAGE_DURATION + 0.2)
    assert b.rage_time == 0 and "rage_end" in kinds(b)


def test_rage_cannot_start_during_transformation():
    b = new()
    b.request_transform()
    assert not b.request_rage()


# ------------------------------------------------------------------ blaster
def test_blaster_charge_fire_damage_and_cooldown():
    b = new()
    hold = Commands(charge_held=True)
    step(b, 0.8, hold)
    assert b.fsm.state == S.CHARGING and 0.3 < b.charge < 0.7
    charge = b.charge
    e_before = b.energy
    step(b, 0.5, Commands())                                 # release -> fire, beam lands
    ev = [e for e in b.events if e.kind == "beam_fire"]
    assert len(ev) == 1 and ev[0].data["hit"]
    assert b.energy < e_before + 5
    hits = [e for e in b.events if e.kind == "enemy_hit"]
    assert hits and hits[0].data["damage"] == pytest.approx(blaster_damage(charge), rel=0.15)
    assert b.enemy_hp < b.enemy_max_hp
    assert b.cooldowns[Ability.BLASTER] >= 0


def test_blaster_tap_below_min_charge_cancels_free():
    b = new()
    b.apply_commands(Commands(charge_held=True))
    b.update(0.03)
    e = b.energy
    b.apply_commands(Commands())
    assert "charge_cancel" in kinds(b) and b.energy == e and b.fsm.state == S.BATTLE_IDLE


def test_held_input_autofires_once_at_full_charge():
    b = new()
    step(b, 4.0, Commands(charge_held=True))
    assert kinds(b).count("beam_fire") == 1                   # latched until released


def test_blaster_needs_energy():
    b = new()
    b.energy = C.BLASTER_MIN_ENERGY - 1
    assert not b.start_charge()
    assert b.fsm.state == S.BATTLE_IDLE


def test_charge_capped_by_available_energy():
    b = new()
    b.energy = 20
    step(b, 3.0, Commands(charge_held=True))
    fire = [e for e in b.events if e.kind == "beam_fire"][0]
    assert fire.data["cost"] <= 20 + 1e-6 and b.energy >= 0


def test_bad_aim_misses_and_resets_combo():
    b = new()
    b.combo = 4
    step(b, 1.0, Commands(charge_held=True, aim=1.0))
    step(b, 0.5, Commands(aim=1.0))
    assert "beam_miss" in kinds(b) and b.combo == 0 and b.enemy_hp == b.enemy_max_hp


def test_aim_is_clamped():
    b = new()
    b.set_aim(50)
    assert b.aim == 1.0
    b.set_aim(-50)
    assert b.aim == -1.0


# ------------------------------------------------------------------ shield
def test_shield_blocks_and_drains_energy():
    b = new()
    step(b, 0.2, Commands(shield_held=True))
    assert b.shield_active and b.fsm.state == S.DEFENDING
    e = b.energy
    step(b, 1.0, Commands(shield_held=True))
    assert b.energy < e - C.SHIELD_DRAIN * 0.8
    b.events.clear()
    taken = b.damage_player(20)
    assert taken == pytest.approx(20 * (1 - C.SHIELD_REDUCTION))
    assert "shield_hit" in kinds(b)


def test_unshielded_hit_does_full_damage_and_resets_combo():
    b = new()
    b.combo = 3
    assert b.damage_player(20) == 20 and b.player_hp == C.PLAYER_MAX_HP - 20 and b.combo == 0


def test_shield_breaks_when_overwhelmed_and_locks():
    b = new()
    step(b, 0.2, Commands(shield_held=True))
    b.shield_strength = 5
    b.damage_player(50)
    assert not b.shield_active and "shield_break" in kinds(b)
    assert b.fsm.state == S.BATTLE_IDLE
    step(b, 0.2, Commands(shield_held=True))
    assert not b.shield_active                                # locked out
    step(b, C.SHIELD_BREAK_LOCK + 2.5, Commands())
    step(b, 0.2, Commands(shield_held=True))
    assert b.shield_active


def test_shield_ends_when_energy_runs_out():
    b = new()
    b.energy = 12
    step(b, 3.0, Commands(shield_held=True))
    assert not b.shield_active and b.energy >= 0


def test_shield_cannot_start_while_charging():
    b = new()
    step(b, 0.3, Commands(charge_held=True, shield_held=True))
    assert b.fsm.state == S.CHARGING and not b.shield_active


# ------------------------------------------------------------------ ultimate
def test_ultimate_requires_energy():
    b = new()
    b.energy = C.ULTIMATE_COST - 1
    assert not b.request_ultimate() and b.fsm.state == S.BATTLE_IDLE
    assert b.can_use(Ability.ULTIMATE)[1] == "energy"


def test_ultimate_sequence_damage_cooldown_and_no_spam():
    b = new()
    b.enemy_hp = b.enemy_max_hp = 10_000
    assert b.request_ultimate() and b.fsm.state == S.ULTIMATE
    assert b.energy == C.PLAYER_MAX_ENERGY - C.ULTIMATE_COST
    step(b, C.ULTIMATE_TOTAL_TIME + 0.5)
    hits = [e for e in b.events if e.kind == "enemy_hit" and e.data["source"] == "ultimate"]
    assert len(hits) == 1 and hits[0].data["damage"] == pytest.approx(C.ULTIMATE_DAMAGE)
    assert b.fsm.state == S.BATTLE_IDLE and b.cooldowns[Ability.ULTIMATE] > 0
    b.energy = 100
    assert not b.request_ultimate()                          # cooldown blocks unlimited ultimates


def test_ultimate_invalid_during_other_actions():
    b = new()
    b.request_transform()
    assert not b.request_ultimate()


def test_ultimate_grants_invulnerability_during_animation():
    b = new()
    b.request_ultimate()
    assert b.damage_player(40) == 0 and b.player_hp == C.PLAYER_MAX_HP


# ------------------------------------------------------------------ energy
def test_energy_regenerates_only_when_idle_and_is_clamped():
    b = new()
    b.energy = 50
    step(b, 2.0)
    assert b.energy == pytest.approx(50 + C.ENERGY_REGEN * 2, abs=0.5)
    step(b, 60)
    assert b.energy == C.PLAYER_MAX_ENERGY
    b.energy = 500
    b.update(DT)
    assert b.energy == C.PLAYER_MAX_ENERGY


def test_training_regenerates_faster():
    t = new(training=True)
    t.energy = 10
    step(t, 1.0)
    assert t.energy > 10 + C.ENERGY_REGEN * 2


# ------------------------------------------------------------------ enemy / endings
def test_enemy_telegraphs_then_attacks_and_player_takes_damage():
    b = Battle(rng=random.Random(2))
    b.enemy_timer = 0.1
    step(b, 0.3)
    assert b.enemy_phase == EnemyPhase.TELEGRAPH and "enemy_telegraph" in kinds(b)
    step(b, b.stats.telegraph + 0.1)
    assert "enemy_attack" in kinds(b)
    step(b, C.ENEMY_PROJECTILE_TIME + 0.2)
    assert b.player_hp == C.PLAYER_MAX_HP - b.stats.damage


def test_strong_hit_staggers_telegraphing_enemy():
    b = Battle(rng=random.Random(2))
    b.enemy_phase, b.enemy_timer = EnemyPhase.TELEGRAPH, 1.0
    b.damage_enemy(C.STAGGER_DAMAGE + 1, "test")
    assert b.enemy_phase == EnemyPhase.RECOVER and "enemy_stagger" in kinds(b)


def test_enemy_defeat_gives_victory_and_score():
    b = new()
    b.enemy_hp = 5
    b.damage_enemy(10, "test")
    b.update(DT)
    assert b.fsm.state == S.VICTORY and b.enemy_phase == EnemyPhase.DEFEATED
    assert "victory" in kinds(b) and b.score >= C.VICTORY_BONUS
    assert b.enemy_hp == 0
    b.update(DT)                                              # no further simulation after victory
    assert b.fsm.state == S.VICTORY


def test_player_defeat():
    b = new()
    b.player_hp = 5
    b.damage_player(30)
    b.update(DT)
    assert b.fsm.state == S.DEFEAT and "defeat" in kinds(b) and b.player_hp == 0


def test_victory_beats_simultaneous_defeat():
    b = new()
    b.enemy_hp, b.player_hp = 0, 0
    b.update(DT)
    assert b.fsm.state == S.VICTORY


def test_next_round_is_harder_and_keeps_score():
    b = new()
    b.enemy_hp = 1
    b.damage_enemy(5, "t")
    b.update(DT)
    score, old_hp = b.score, b.enemy_max_hp
    b.next_round()
    assert b.round == 2 and b.enemy_max_hp > old_hp and b.score == score
    assert b.fsm.state == S.BATTLE_IDLE and b.upgrades == 1
    assert b.power > power_level(100)


def test_restart_resets_everything():
    b = new()
    b.request_transform()
    b.score, b.player_hp, b.round = 999, 10, 4
    b.restart()
    assert (b.score, b.player_hp, b.round, b.transformed) == (0, C.PLAYER_MAX_HP, 1, False)
    assert b.fsm.state == S.BATTLE_IDLE and b.energy == C.PLAYER_MAX_ENERGY
    assert all(v == 0 for v in b.cooldowns.values())


def test_combo_increments_and_times_out():
    b = new()
    b.damage_enemy(1, "t")
    b.damage_enemy(1, "t")
    assert b.combo == 2
    step(b, C.COMBO_WINDOW + 0.2)
    assert b.combo == 0


def test_training_dummy_never_dies_and_player_never_hurt():
    b = new(training=True)
    b.enemy_hp = 1
    b.damage_enemy(50, "t")
    b.update(DT)
    assert b.fsm.state == S.TRAINING and b.enemy_hp == b.enemy_max_hp and "dummy_reset" in kinds(b)
    assert b.damage_player(50) == 0 and b.player_hp == C.PLAYER_MAX_HP


# ------------------------------------------------------------------ pause / validity
def test_pause_freezes_simulation_and_ignores_input():
    b = new()
    b.request_transform()
    b.fsm.pause()
    snapshot = (b.anim_time, b.energy, dict(b.cooldowns))
    for _ in range(120):
        b.apply_commands(Commands(charge_held=True, scan=True))
        b.update(DT)
    assert (b.anim_time, b.energy, dict(b.cooldowns)) == snapshot
    assert b.fsm.resume() and b.fsm.state == S.TRANSFORMING
    step(b, C.TRANSFORM_ANIM + 0.1)
    assert b.transformed


def test_no_actions_after_battle_ends():
    b = new()
    b.fsm.transition(S.VICTORY)
    assert not b.request_transform() and not b.start_charge() and not b.request_scan()
    assert b.can_use(Ability.SHIELD) == (False, "not_in_battle")


def test_scanner_energy_cooldown_and_duration():
    b = new()
    assert b.request_scan() and b.scan_time == C.SCAN_DURATION
    assert not b.request_scan()                               # already active
    step(b, C.SCAN_DURATION + 0.2)
    assert b.scan_time == 0 and not b.request_scan()          # cooldown
    b.energy = 1
    b.cooldowns[Ability.SCANNER] = 0
    assert not b.request_scan()


def test_update_with_invalid_dt_is_safe():
    b = new()
    b.update(0)
    b.update(-1)
    b.update(1e9)                                              # clamped internally
    assert 0 <= b.energy <= C.PLAYER_MAX_ENERGY and b.player_hp > 0


def test_shared_fsm_between_app_and_battle():
    fsm = StateMachine(S.MENU)
    fsm.transition(S.TRAINING)
    b = Battle(fsm=fsm, training=True)
    b.request_rage()
    assert fsm.state == S.RAGE_BURST
