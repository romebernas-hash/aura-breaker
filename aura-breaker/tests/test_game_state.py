import pytest

from game_state import ACTION_STATES, Commands, GameState as S, StateMachine


def test_menu_to_battle_and_back():
    fsm = StateMachine(S.MENU)
    assert fsm.transition(S.BATTLE_IDLE)
    assert fsm.transition(S.MENU)


def test_invalid_transitions_rejected():
    fsm = StateMachine(S.MENU)
    assert not fsm.transition(S.CHARGING)          # cannot attack from menu
    assert fsm.state == S.MENU
    assert not fsm.transition(S.MENU)              # same state is a no-op


def test_transformation_state_transitions():
    fsm = StateMachine(S.BATTLE_IDLE)
    assert fsm.transition(S.TRANSFORMING)
    # incompatible actions cannot start during the transformation
    for s in (S.CHARGING, S.DEFENDING, S.RAGE_BURST, S.ULTIMATE, S.ATTACKING):
        assert not fsm.can_transition(s)
    assert fsm.transition(S.BATTLE_IDLE)


@pytest.mark.parametrize("a", sorted(ACTION_STATES, key=lambda s: s.name))
def test_actions_are_mutually_exclusive(a):
    fsm = StateMachine(S.BATTLE_IDLE)
    assert fsm.transition(a)
    others = ACTION_STATES - {a}
    allowed = {s for s in others if fsm.can_transition(s)}
    assert allowed <= {S.ATTACKING} and (allowed == set() or a == S.CHARGING)


def test_charging_leads_to_attacking():
    fsm = StateMachine(S.TRAINING)
    fsm.transition(S.CHARGING)
    assert fsm.transition(S.ATTACKING)
    assert fsm.transition(S.TRAINING)


def test_pause_and_resume_restore_previous_state():
    fsm = StateMachine(S.BATTLE_IDLE)
    fsm.transition(S.CHARGING)
    assert fsm.pause() and fsm.state == S.PAUSED
    assert fsm.resume() and fsm.state == S.CHARGING
    assert not fsm.resume()


def test_cannot_pause_outside_combat():
    for s in (S.MENU, S.SETTINGS, S.CALIBRATION, S.VICTORY, S.DEFEAT):
        assert not StateMachine(s).pause()


def test_end_states_only_exit_to_idle_or_menu():
    for end in (S.VICTORY, S.DEFEAT):
        fsm = StateMachine(S.BATTLE_IDLE)
        fsm.transition(end)
        assert not fsm.can_transition(S.CHARGING)
        assert fsm.can_transition(S.BATTLE_IDLE) and fsm.can_transition(S.MENU)


def test_force_resets_pause():
    fsm = StateMachine(S.BATTLE_IDLE)
    fsm.pause()
    fsm.force(S.BATTLE_IDLE)
    assert fsm.resume_state is None and fsm.state == S.BATTLE_IDLE


def test_history_is_bounded():
    fsm = StateMachine(S.BATTLE_IDLE)
    for _ in range(200):
        fsm.transition(S.CHARGING)
        fsm.transition(S.BATTLE_IDLE)
    assert len(fsm.history) <= 64


def test_commands_merge():
    a = Commands(charge_held=True, aim=0.8)
    b = Commands(shield_held=True, aim=0.7, scan=True)
    m = a.merged(b)
    assert m.charge_held and m.shield_held and m.scan
    assert m.aim == 1.0   # clamped
