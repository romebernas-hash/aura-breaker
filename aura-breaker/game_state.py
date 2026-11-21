"""Game finite-state machine and the input command container.

Every screen and every combat phase is a :class:`GameState`. Only transitions
listed in :data:`VALID_TRANSITIONS` are permitted, which is what stops
incompatible abilities from running at the same time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

log = logging.getLogger("aura.state")


class GameState(Enum):
    MENU = auto()
    SETTINGS = auto()
    CALIBRATION = auto()
    TRAINING = auto()
    BATTLE_IDLE = auto()
    CHARGING = auto()
    TRANSFORMING = auto()
    ATTACKING = auto()
    DEFENDING = auto()
    RAGE_BURST = auto()
    ULTIMATE = auto()
    VICTORY = auto()
    DEFEAT = auto()
    PAUSED = auto()


S = GameState
IDLE_STATES = frozenset({S.BATTLE_IDLE, S.TRAINING})
ACTION_STATES = frozenset({S.CHARGING, S.TRANSFORMING, S.ATTACKING, S.DEFENDING, S.RAGE_BURST, S.ULTIMATE})
COMBAT_STATES = IDLE_STATES | ACTION_STATES
PAUSABLE_STATES = COMBAT_STATES
END_STATES = frozenset({S.VICTORY, S.DEFEAT})

_common_exit = {S.VICTORY, S.DEFEAT, S.PAUSED, S.MENU}

VALID_TRANSITIONS: dict[GameState, frozenset[GameState]] = {
    S.MENU: frozenset({S.CALIBRATION, S.TRAINING, S.BATTLE_IDLE, S.SETTINGS}),
    S.SETTINGS: frozenset({S.MENU, S.CALIBRATION}),
    S.CALIBRATION: frozenset({S.MENU, S.SETTINGS}),
    # Idle states can start any single action.
    S.BATTLE_IDLE: frozenset(ACTION_STATES | _common_exit),
    S.TRAINING: frozenset(ACTION_STATES | _common_exit),
    # Actions are mutually exclusive: they return to idle (or CHARGING -> ATTACKING).
    S.CHARGING: frozenset({S.ATTACKING} | IDLE_STATES | _common_exit),
    S.ATTACKING: frozenset(IDLE_STATES | _common_exit),
    S.DEFENDING: frozenset(IDLE_STATES | _common_exit),
    S.TRANSFORMING: frozenset(IDLE_STATES | _common_exit),
    S.RAGE_BURST: frozenset(IDLE_STATES | _common_exit),
    S.ULTIMATE: frozenset(IDLE_STATES | _common_exit),
    S.VICTORY: frozenset(IDLE_STATES | {S.MENU}),
    S.DEFEAT: frozenset(IDLE_STATES | {S.MENU}),
    S.PAUSED: frozenset(COMBAT_STATES | {S.MENU}),
}


class StateMachine:
    """Small validated FSM with pause/resume support."""

    def __init__(self, initial: GameState = S.MENU) -> None:
        self.state: GameState = initial
        self.history: list[GameState] = [initial]
        self._resume_state: GameState | None = None

    # ------------------------------------------------------------------
    def can_transition(self, new: GameState) -> bool:
        return new in VALID_TRANSITIONS.get(self.state, frozenset())

    def transition(self, new: GameState) -> bool:
        """Move to ``new`` if legal. Returns True on success."""
        if new == self.state:
            return False
        if not self.can_transition(new):
            log.debug("Rejected transition %s -> %s", self.state.name, new.name)
            return False
        self.state = new
        self.history.append(new)
        if len(self.history) > 64:
            del self.history[:32]
        return True

    # -- pause -----------------------------------------------------------
    def pause(self) -> bool:
        if self.state not in PAUSABLE_STATES:
            return False
        resume = self.state
        if self.transition(S.PAUSED):
            self._resume_state = resume
            return True
        return False

    def resume(self) -> bool:
        if self.state != S.PAUSED or self._resume_state is None:
            return False
        target, self._resume_state = self._resume_state, None
        return self.transition(target)

    # -- helpers -----------------------------------------------------------
    @property
    def in_combat(self) -> bool:
        return self.state in COMBAT_STATES

    @property
    def is_idle(self) -> bool:
        return self.state in IDLE_STATES

    @property
    def resume_state(self) -> GameState | None:
        return self._resume_state

    def force(self, new: GameState) -> None:
        """Hard reset used only by restart / menu flows (bypasses validation)."""
        self.state = new
        self._resume_state = None
        self.history.append(new)


@dataclass
class Commands:
    """Abstract player input, produced by keyboard and by face gestures."""

    charge_held: bool = False
    shield_held: bool = False
    brow_action: bool = False   # edge: transformation / rage depending on priority
    scan: bool = False          # edge
    ultimate: bool = False      # edge
    transform: bool = False     # edge (keyboard 1)
    rage: bool = False          # edge (keyboard 4)
    aim: float = 0.0            # -1..1, positive = up

    def merged(self, other: "Commands") -> "Commands":
        return Commands(
            charge_held=self.charge_held or other.charge_held,
            shield_held=self.shield_held or other.shield_held,
            brow_action=self.brow_action or other.brow_action,
            scan=self.scan or other.scan,
            ultimate=self.ultimate or other.ultimate,
            transform=self.transform or other.transform,
            rage=self.rage or other.rage,
            aim=max(-1.0, min(1.0, self.aim + other.aim)),
        )
