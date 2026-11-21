"""Face landmarks -> normalized measurements -> gestures -> game commands.

Everything here is pure NumPy/Python (no camera, no MediaPipe import) so it can
be unit tested with simulated measurements.

Important: gestures are *geometric* measurements (mouth gap, brow-to-eye
distance, lip width, head roll). They are not emotion recognition, and each
player's face is different, hence the calibration step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

import config as C
from game_state import Commands


class Gesture(Enum):
    MOUTH_OPEN = auto()
    BROW_RAISE = auto()
    BROW_LOWER = auto()
    PURSE = auto()


GESTURE_SIGNAL = {
    Gesture.MOUTH_OPEN: "mouth_open",
    Gesture.BROW_RAISE: "brow_raise",
    Gesture.BROW_LOWER: "brow_raise",
    Gesture.PURSE: "lip_width",
}
GESTURE_LABEL = {
    Gesture.MOUTH_OPEN: "MOUTH OPEN",
    Gesture.BROW_RAISE: "BROWS UP",
    Gesture.BROW_LOWER: "BROWS DOWN",
    Gesture.PURSE: "LIPS PURSED",
}
_EXCLUSIVE = {
    Gesture.BROW_RAISE: Gesture.BROW_LOWER,
    Gesture.BROW_LOWER: Gesture.BROW_RAISE,
    Gesture.MOUTH_OPEN: Gesture.PURSE,
    Gesture.PURSE: Gesture.MOUTH_OPEN,
}

# MediaPipe Face Mesh landmark indices used below.
LM_FOREHEAD, LM_CHIN = 10, 152
LM_CHEEK_L, LM_CHEEK_R = 234, 454
LM_LIP_TOP, LM_LIP_BOTTOM = 13, 14
LM_MOUTH_L, LM_MOUTH_R = 61, 291
LM_BROW_L, LM_BROW_R = 105, 334
LM_EYE_TOP_L, LM_EYE_TOP_R = 159, 386
LM_EYE_OUT_A, LM_EYE_OUT_B = 33, 263
LM_NOSE = 1
MIN_LANDMARKS = 455


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
@dataclass
class FaceMeasurements:
    mouth_open: float      # inner-lip gap / face height
    brow_raise: float      # mean brow-to-upper-eyelid distance / face height
    lip_width: float       # mouth corner distance / face width
    roll_deg: float        # >0: head tilted toward the player's LEFT shoulder
    nose_x: float = 0.5
    nose_y: float = 0.5


def measure_landmarks(lm, aspect: float = 1.0) -> FaceMeasurements | None:
    """Compute normalized measurements from (N, >=2) normalized landmarks.

    ``aspect`` is frame width / height so distances are measured in true
    proportions. Returns None for malformed or degenerate input.
    """
    try:
        arr = np.asarray(lm, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < MIN_LANDMARKS or arr.shape[1] < 2:
        return None
    pts = arr[:, :2].copy()
    if not np.all(np.isfinite(pts)):
        return None
    pts[:, 0] *= aspect

    def d(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    face_h = d(LM_FOREHEAD, LM_CHIN)
    face_w = d(LM_CHEEK_L, LM_CHEEK_R)
    if face_h < 1e-6 or face_w < 1e-6:
        return None
    mouth = d(LM_LIP_TOP, LM_LIP_BOTTOM) / face_h
    brow = 0.5 * (d(LM_BROW_L, LM_EYE_TOP_L) + d(LM_BROW_R, LM_EYE_TOP_R)) / face_h
    width = d(LM_MOUTH_L, LM_MOUTH_R) / face_w
    a, b = pts[LM_EYE_OUT_A], pts[LM_EYE_OUT_B]
    if a[0] > b[0]:
        a, b = b, a
    roll = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    return FaceMeasurements(mouth, brow, width, roll, float(arr[LM_NOSE, 0]), float(arr[LM_NOSE, 1]))


# --------------------------------------------------------------------------
# Calibration data
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    baseline: dict[str, float] | None = None
    deltas: dict[str, float] = field(default_factory=lambda: dict(C.DEFAULT_DELTAS))
    calibrated: bool = False
    warnings: list[str] = field(default_factory=list)

    def score(self, gesture: Gesture, m: FaceMeasurements, baseline: dict[str, float] | None = None) -> float:
        """0 = neutral, 1 = the calibrated full gesture, up to 1.5."""
        base = baseline or self.baseline or C.DEFAULT_BASELINE
        sig = GESTURE_SIGNAL[gesture]
        delta = self.deltas.get(gesture.name, C.DEFAULT_DELTAS[gesture.name])
        if abs(delta) < 1e-9:
            return 0.0
        return clamp((getattr(m, sig) - base[sig]) / delta, 0.0, 1.5)

    def to_dict(self) -> dict:
        return {"baseline": self.baseline, "deltas": dict(self.deltas), "calibrated": self.calibrated}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Calibration":
        if not data:
            return cls()
        try:
            deltas = dict(C.DEFAULT_DELTAS)
            deltas.update({k: float(v) for k, v in data.get("deltas", {}).items() if k in deltas})
            base = data.get("baseline")
            if base is not None:
                base = {k: float(base[k]) for k in C.DEFAULT_BASELINE}
            return cls(base, deltas, bool(data.get("calibrated", False)))
        except (KeyError, TypeError, ValueError):
            return cls()


# --------------------------------------------------------------------------
# Gesture detection
# --------------------------------------------------------------------------
@dataclass
class _GState:
    score: float = 0.0
    active: bool = False
    above: float = 0.0
    hold: float = 0.0
    cooldown: float = 0.0


@dataclass
class GestureFrame:
    face_present: bool
    tracking_ready: bool
    neutral: bool
    scores: dict[Gesture, float]
    active: set[Gesture]
    pressed: list[Gesture]
    released: list[Gesture]
    hold: dict[Gesture, float]
    aim: float


class GestureDetector:
    """Hysteresis + hold-time + cooldown + neutral-gating gesture detector."""

    def __init__(
        self,
        calibration: Calibration | None = None,
        sensitivity: float = 1.0,
        hold_time: float = 0.12,
        invert_tilt: bool = False,
    ) -> None:
        self.cal = calibration or Calibration()
        self.sensitivity = sensitivity
        self.hold_time = hold_time
        self.invert_tilt = invert_tilt
        self.reset()

    def reset(self) -> None:
        self.states = {g: _GState() for g in Gesture}
        self.baseline = dict(self.cal.baseline) if self.cal.baseline else None
        self._auto: list[FaceMeasurements] = []
        self._auto_time = 0.0
        self.neutral_seen = False
        self._neutral_time = 0.0
        self._no_face = 0.0
        self.face_present = False
        self.aim = 0.0

    def set_calibration(self, cal: Calibration) -> None:
        self.cal = cal
        self.reset()

    @property
    def tracking_ready(self) -> bool:
        return self.baseline is not None

    # ------------------------------------------------------------------
    def _release_all(self, released: list[Gesture]) -> None:
        for g, st in self.states.items():
            if st.active:
                released.append(g)
            self.states[g] = _GState()
        self.neutral_seen = False
        self._neutral_time = 0.0
        self.aim = 0.0

    def _frame(self, pressed, released, present: bool) -> GestureFrame:
        return GestureFrame(
            face_present=present,
            tracking_ready=self.tracking_ready,
            neutral=self.neutral_seen,
            scores={g: s.score for g, s in self.states.items()},
            active={g for g, s in self.states.items() if s.active},
            pressed=pressed,
            released=released,
            hold={g: s.hold for g, s in self.states.items()},
            aim=self.aim,
        )

    def _conflicts(self, g: Gesture) -> bool:
        other = self.states[_EXCLUSIVE[g]]
        return other.active or other.score > self.states[g].score

    def update(self, meas: FaceMeasurements | None, dt: float) -> GestureFrame:
        pressed: list[Gesture] = []
        released: list[Gesture] = []
        dt = max(0.0, dt)

        if meas is None:
            self._no_face += dt
            if self._no_face > C.FACE_LOSS_GRACE and self.face_present:
                self._release_all(released)
                self.face_present = False
            return self._frame(pressed, released, self.face_present)
        self._no_face = 0.0
        self.face_present = True

        # Auto-baseline when the player has not calibrated yet.
        if self.baseline is None:
            self._auto.append(meas)
            self._auto_time += dt
            if self._auto_time >= C.AUTO_BASELINE_SECONDS and len(self._auto) >= 3:
                self.baseline = {
                    "mouth_open": float(np.median([m.mouth_open for m in self._auto])),
                    "brow_raise": float(np.median([m.brow_raise for m in self._auto])),
                    "lip_width": float(np.median([m.lip_width for m in self._auto])),
                    "roll_deg": float(np.median([m.roll_deg for m in self._auto])),
                }
            return self._frame(pressed, released, True)

        alpha = 1.0 - math.exp(-dt / C.GESTURE_SMOOTH_TAU) if dt > 0 else 0.0
        for g, st in self.states.items():
            raw = self.cal.score(g, meas, self.baseline) * self.sensitivity
            st.score += (clamp(raw, 0.0, 1.5) - st.score) * alpha

        if all(st.score < C.GESTURE_NEUTRAL for st in self.states.values()):
            self._neutral_time += dt
            if self._neutral_time >= 0.1:
                self.neutral_seen = True
        else:
            self._neutral_time = 0.0

        for g, st in self.states.items():
            if st.active:
                st.hold += dt
                if st.score < C.GESTURE_OFF:
                    st.active = False
                    st.hold = 0.0
                    st.above = 0.0
                    st.cooldown = C.GESTURE_RELEASE_COOLDOWN
                    released.append(g)
            else:
                st.cooldown = max(0.0, st.cooldown - dt)
                if self.neutral_seen and st.score >= C.GESTURE_ON and st.cooldown <= 0 and not self._conflicts(g):
                    st.above += dt
                    if st.above >= self.hold_time:
                        st.active = True
                        st.hold = 0.0
                        pressed.append(g)
                else:
                    st.above = 0.0

        # Aim from head roll relative to the calibrated neutral roll.
        rel = meas.roll_deg - self.baseline.get("roll_deg", 0.0)
        mag = max(0.0, abs(rel) - C.AIM_DEADZONE_DEG)
        target = math.copysign(mag, rel) / max(1e-6, C.AIM_ROLL_RANGE_DEG - C.AIM_DEADZONE_DEG)
        target = clamp(target * self.sensitivity, -1.0, 1.0)
        if self.invert_tilt:
            target = -target
        a2 = 1.0 - math.exp(-dt / 0.08) if dt > 0 else 0.0
        self.aim += (target - self.aim) * a2
        return self._frame(pressed, released, True)


# --------------------------------------------------------------------------
# Gesture -> command mapping
# --------------------------------------------------------------------------
class FaceInputMapper:
    """Turns gesture frames into edge/level commands for the battle system.

    Mapping (all configurable via calibration/sensitivity):
        BROW_LOWER pressed                -> brow action (transformation / rage)
        MOUTH_OPEN held                   -> charge; release fires
        BROW_RAISE held (mouth closed)    -> power scanner (once per hold)
        BROW_RAISE + MOUTH_OPEN together  -> ultimate sequence (held ULTIMATE_HOLD s)
        PURSE held                        -> shield
        head roll                         -> aim
    """

    def __init__(self) -> None:
        self._scan_fired = False
        self._ult_fired = False
        self._combo = 0.0

    def reset(self) -> None:
        self._scan_fired = False
        self._ult_fired = False
        self._combo = 0.0

    def map(self, frame: GestureFrame, dt: float) -> Commands:
        cmds = Commands()
        if not frame.face_present or not frame.tracking_ready:
            self.reset()
            return cmds
        act = frame.active
        raise_ = Gesture.BROW_RAISE in act
        mouth = Gesture.MOUTH_OPEN in act

        if raise_ and mouth:
            self._combo += dt
            if self._combo >= C.ULTIMATE_HOLD and not self._ult_fired:
                cmds.ultimate = True
                self._ult_fired = True
        else:
            self._combo = 0.0
            if not (raise_ or mouth):
                self._ult_fired = False

        if raise_ and not mouth:
            if frame.hold.get(Gesture.BROW_RAISE, 0.0) >= C.SCAN_HOLD and not self._scan_fired:
                cmds.scan = True
                self._scan_fired = True
        elif not raise_:
            self._scan_fired = False

        cmds.brow_action = Gesture.BROW_LOWER in frame.pressed
        cmds.charge_held = mouth and not raise_
        cmds.shield_held = Gesture.PURSE in act
        cmds.aim = frame.aim
        return cmds


# --------------------------------------------------------------------------
# Calibration wizard (pure logic; the UI just displays its state)
# --------------------------------------------------------------------------
class CalibStep(Enum):
    NEUTRAL = auto()
    MOUTH_OPEN = auto()
    BROW_RAISE = auto()
    BROW_LOWER = auto()
    PURSE = auto()
    TEST = auto()


CALIB_PROMPTS = {
    CalibStep.NEUTRAL: ("NEUTRAL FACE", "Look at the camera with a relaxed, neutral face."),
    CalibStep.MOUTH_OPEN: ("OPEN YOUR MOUTH", "Open your mouth wide, like a battle cry."),
    CalibStep.BROW_RAISE: ("RAISE YOUR EYEBROWS", "Lift both eyebrows as high as you comfortably can."),
    CalibStep.BROW_LOWER: ("INTENSE BROWS", "Pull your eyebrows down into a fierce scowl."),
    CalibStep.PURSE: ("SHIELD GESTURE", "Purse your lips together, as if saying 'oo'."),
    CalibStep.TEST: ("TEST YOUR GESTURES", "Try each gesture. ENTER saves, R repeats calibration."),
}


class Calibrator:
    """Walks the player through neutral + four gestures and builds a Calibration."""

    def __init__(self, prep: float = C.CALIB_PREP_TIME, record: float = C.CALIB_RECORD_TIME) -> None:
        self.prep_time = prep
        self.record_time = record
        self.steps = list(CalibStep)
        self.index = 0
        self.phase = "prep"      # prep | record
        self.t = 0.0
        self.samples: list[FaceMeasurements] = []
        self.baseline: dict[str, float] | None = None
        self.deltas: dict[str, float] = dict(C.DEFAULT_DELTAS)
        self.warnings: list[str] = []
        self.no_face_time = 0.0

    @property
    def step(self) -> CalibStep:
        return self.steps[min(self.index, len(self.steps) - 1)]

    @property
    def finished(self) -> bool:
        return self.step == CalibStep.TEST

    @property
    def progress(self) -> float:
        total = self.prep_time if self.phase == "prep" else self.record_time
        return clamp(self.t / total, 0.0, 1.0) if total > 0 else 1.0

    def restart(self) -> None:
        self.__init__(self.prep_time, self.record_time)

    def update(self, meas: FaceMeasurements | None, dt: float) -> None:
        if self.finished:
            return
        self.no_face_time = 0.0 if meas is not None else self.no_face_time + dt
        if self.phase == "prep":
            self.t += dt
            if self.t >= self.prep_time:
                self.phase, self.t, self.samples = "record", 0.0, []
            return
        if meas is not None:  # recording time only advances while a face is visible
            self.t += dt
            self.samples.append(meas)
        if self.t >= self.record_time:
            self._finish_step()

    def _finish_step(self) -> None:
        step = self.step
        samples = self.samples
        if step == CalibStep.NEUTRAL:
            if len(samples) >= 5:
                self.baseline = {
                    k: float(np.median([getattr(m, k) for m in samples])) for k in C.DEFAULT_BASELINE
                }
            else:
                self.baseline = dict(C.DEFAULT_BASELINE)
                self.warnings.append("Neutral face: not enough data, using defaults.")
        else:
            g = Gesture[step.name]
            sig = GESTURE_SIGNAL[g]
            base = (self.baseline or C.DEFAULT_BASELINE)[sig]
            default = C.DEFAULT_DELTAS[g.name]
            if len(samples) >= 5:
                vals = np.array([getattr(m, sig) for m in samples])
                peak = float(np.percentile(vals, 90 if default > 0 else 10))
                delta = peak - base
                ok = delta * default > 0 and abs(delta) >= C.MIN_SEPARATION[g.name]
            else:
                delta, ok = 0.0, False
            if ok:
                self.deltas[g.name] = delta
            else:
                self.warnings.append(
                    f"{GESTURE_LABEL[g]}: movement too small to measure - using default threshold. "
                    "Try again or raise sensitivity in Settings."
                )
        self.index += 1
        self.phase, self.t, self.samples = "prep", 0.0, []

    def build(self) -> Calibration:
        return Calibration(
            baseline=dict(self.baseline or C.DEFAULT_BASELINE),
            deltas=dict(self.deltas),
            calibrated=self.baseline is not None,
            warnings=list(self.warnings),
        )
