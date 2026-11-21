"""Hand gesture recognition and the hand -> combat-input state machine.

Pure Python + numpy (no pygame, no camera, no MediaPipe) so it can be unit tested.

Gesture design (all thresholds live in :class:`HandConfig`):

* KI BLAST  - hold an OPEN PALM to charge. Close the hand (fist) or lower it to fire.
* SHIELD    - hold a CLOSED FIST for ``shield_hold`` seconds. A fist that you make
              right after charging (the "fire" fist) never raises the shield; you must
              open the hand again first.
* ULTIMATE  - bring BOTH HANDS together (Kamehameha pose) and hold ``ultimate_hold``
              seconds. Fires once, then needs the hands to separate before re-arming.
* Tracking loss while charging cancels the charge (it does NOT fire) after
  ``loss_grace`` seconds, so a dropped frame never causes an accidental shot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
MCP = (5, 9, 13, 17)     # index, middle, ring, pinky knuckles
PIP = (6, 10, 14, 18)
TIP = (8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)


class HandPose(Enum):
    NONE = "none"
    OPEN = "open palm"
    FIST = "fist"
    OTHER = "other"


@dataclass
class HandObs:
    """One detected hand. ``lm`` is (21, 3) normalized landmarks in the raw frame."""
    lm: np.ndarray
    score: float = 1.0          # detector / handedness confidence 0..1
    label: str = ""             # "Left" / "Right" as reported (unreliable, informational)


@dataclass
class HandConfig:
    # pose classification: fingertip-to-wrist distance / knuckle-to-wrist distance
    extended_ratio: float = 1.50
    curled_ratio: float = 1.30
    min_confidence: float = 0.50
    pose_stable: float = 0.08            # seconds a new pose must persist before it is accepted
    # ki blast
    charge_start_delay: float = 0.10     # open palm must be stable this long
    other_grace: float = 0.40            # tolerate OTHER pose flicker while charging
    loss_grace: float = 0.50             # tracking loss tolerated while charging; then cancel
    # shield
    shield_hold: float = 0.35
    shield_release_grace: float = 0.15
    fire_fist_window: float = 0.50       # OPEN -> FIST within this = "fire fist", not shield
    # ultimate
    ultimate_distance: float = 2.6       # max distance between palm centres, in hand-size units
    ultimate_hold: float = 0.60
    ultimate_cooldown: float = 1.50
    ultimate_loss_grace: float = 0.25


@dataclass
class HandFrame:
    """Result of one update."""
    charge_held: bool = False
    shield_held: bool = False
    ultimate: bool = False           # edge: True for exactly one frame
    cancel_charge: bool = False      # edge: drop the charge without firing
    pose: HandPose = HandPose.NONE
    confidence: float = 0.0
    hands: int = 0
    action: str = ""                 # short UI text, e.g. "CHARGING KI"
    message: str = ""                # hint / warning text
    ultimate_progress: float = 0.0   # 0..1 hold progress
    aim_point: tuple[float, float] | None = None   # normalized raw-frame position of the active hand


# --------------------------------------------------------------------------
# Geometry / classification
# --------------------------------------------------------------------------
def _xy(lm: np.ndarray, aspect: float) -> np.ndarray:
    """Landmarks as 2D points in aspect-corrected units (x scaled by w/h)."""
    pts = np.asarray(lm, dtype=np.float64)[:, :2].copy()
    pts[:, 0] *= aspect
    return pts


def hand_size(lm: np.ndarray, aspect: float = 16 / 9) -> float:
    p = _xy(lm, aspect)
    return float(np.linalg.norm(p[9] - p[WRIST])) or 1e-6


def palm_center(lm: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(lm, dtype=np.float64)[list(PALM_POINTS), :2]
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def finger_ratios(lm: np.ndarray, aspect: float = 16 / 9) -> list[float]:
    """For index..pinky: |tip - wrist| / |mcp - wrist|. ~2 when extended, ~1 when curled."""
    p = _xy(lm, aspect)
    w = p[WRIST]
    out = []
    for m, t in zip(MCP, TIP):
        d_m = np.linalg.norm(p[m] - w) or 1e-6
        out.append(float(np.linalg.norm(p[t] - w) / d_m))
    return out


def classify_pose(lm: np.ndarray, cfg: HandConfig | None = None, aspect: float = 16 / 9) -> tuple[HandPose, float]:
    """Return (pose, pose_confidence 0..1) for one hand."""
    cfg = cfg or HandConfig()
    ratios = finger_ratios(lm, aspect)
    ext = [r > cfg.extended_ratio for r in ratios]
    cur = [r < cfg.curled_ratio for r in ratios]
    span = max(1e-6, cfg.extended_ratio - cfg.curled_ratio)
    if sum(ext) >= 3 and sum(cur) == 0:
        conf = float(np.mean([min(1.0, max(0.0, (r - cfg.curled_ratio) / span)) for r in ratios]))
        return HandPose.OPEN, conf
    if sum(cur) >= 3 and sum(ext) == 0:
        conf = float(np.mean([min(1.0, max(0.0, (cfg.extended_ratio - r) / span)) for r in ratios]))
        return HandPose.FIST, conf
    return HandPose.OTHER, 0.3


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------
class HandGestureEngine:
    """Turns per-frame hand observations into debounced combat inputs."""

    def __init__(self, cfg: HandConfig | None = None, aspect: float = 16 / 9) -> None:
        self.cfg = cfg or HandConfig()
        self.aspect = aspect
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.pose = HandPose.NONE            # debounced pose of the primary hand
        self._cand = HandPose.NONE
        self._cand_t = 0.0
        self._open_t = 0.0                   # how long the debounced pose has been OPEN
        self._fist_t = 0.0
        self._since_open = 99.0              # time since the pose was last OPEN
        self._fist_blocked = False           # "fire fist" must be released before shield can arm
        self._shield_on = False
        self._shield_gap = 0.0
        self._charging = False
        self._other_t = 0.0
        self._loss_t = 0.0                   # time with no hands at all
        self._ult_t = 0.0
        self._ult_gap = 0.0
        self._ult_latched = False
        self._ult_cd = 0.0
        self._last_aim: tuple[float, float] | None = None
        self._last_aim_age = 99.0

    @property
    def aim_age(self) -> float:
        return self._last_aim_age

    # -- helpers ---------------------------------------------------------
    def _debounce(self, raw: HandPose, dt: float) -> HandPose:
        if raw == self.pose:
            self._cand, self._cand_t = raw, 0.0
            return self.pose
        if raw != self._cand:
            self._cand, self._cand_t = raw, 0.0
        self._cand_t += dt
        if self._cand_t >= self.cfg.pose_stable or self.pose == HandPose.NONE:
            self.pose = raw
            self._cand_t = 0.0
        return self.pose

    def _usable(self, hands: list[HandObs]) -> list[HandObs]:
        return [h for h in hands if h.lm is not None and len(h.lm) >= 21 and h.score >= self.cfg.min_confidence]

    # -- main update -----------------------------------------------------
    def update(self, hands: list[HandObs] | None, dt: float) -> HandFrame:
        cfg = self.cfg
        dt = max(0.0, min(dt, 0.05))   # a frame hitch must never skip a debounce window
        usable = self._usable(hands or [])
        out = HandFrame(hands=len(usable))
        self._ult_cd = max(0.0, self._ult_cd - dt)
        self._last_aim_age += dt

        # ---- no hands: grace, then cancel (never fire) -------------------
        if not usable:
            self._loss_t += dt
            self._ult_gap += dt
            if self._ult_gap > cfg.ultimate_loss_grace:
                self._ult_t = 0.0
                self._ult_latched = False if self._ult_cd <= 0 else self._ult_latched
            if self._charging and self._loss_t > cfg.loss_grace:
                self._charging = False
                out.cancel_charge = True
                out.message = "HAND LOST - CHARGE CANCELLED"
            if self._loss_t > cfg.shield_release_grace:
                self._shield_on = False
                self._fist_t = 0.0
            self.pose = HandPose.NONE if self._loss_t > cfg.loss_grace else self.pose
            out.charge_held = self._charging
            out.shield_held = self._shield_on
            if self._charging:
                out.action = "CHARGING KI"
            out.pose = self.pose if self._loss_t <= cfg.loss_grace else HandPose.NONE
            out.aim_point = self._last_aim
            return out
        self._loss_t = 0.0

        # ---- per-hand classification ------------------------------------
        info = []
        for h in usable:
            pose, pconf = classify_pose(h.lm, cfg, self.aspect)
            info.append((h, pose, pconf * h.score))
        info.sort(key=lambda t: t[2], reverse=True)
        prim, raw_pose, conf = info[0]
        out.confidence = conf
        cx, cy = palm_center(prim.lm)

        # ---- ultimate: both hands close together -------------------------
        ult_candidate = False
        if len(usable) >= 2:
            a, b = usable[0], usable[1]
            pa = _xy(np.array([palm_center(a.lm) + (0.0,)]), self.aspect)[0]
            pb = _xy(np.array([palm_center(b.lm) + (0.0,)]), self.aspect)[0]
            size = 0.5 * (hand_size(a.lm, self.aspect) + hand_size(b.lm, self.aspect))
            dist = float(np.linalg.norm(pa - pb)) / size
            poses = [classify_pose(h.lm, cfg, self.aspect)[0] for h in (a, b)]
            both_fists = all(p == HandPose.FIST for p in poses)
            if dist <= cfg.ultimate_distance and not both_fists:
                ult_candidate = True
                mx, my = (palm_center(a.lm)[0] + palm_center(b.lm)[0]) / 2, (palm_center(a.lm)[1] + palm_center(b.lm)[1]) / 2
                cx, cy = mx, my
        if ult_candidate:
            self._ult_gap = 0.0
            if self._charging:  # approaching with the second hand must not fire a blast
                self._charging = False
                out.cancel_charge = True
            if not self._ult_latched and self._ult_cd <= 0:
                self._ult_t += dt
                out.ultimate_progress = min(1.0, self._ult_t / cfg.ultimate_hold)
                out.action = "ULTIMATE CHARGING"
                if self._ult_t >= cfg.ultimate_hold:
                    out.ultimate = True
                    self._ult_latched = True
                    self._ult_cd = cfg.ultimate_cooldown
                    self._ult_t = 0.0
                    out.action = "ULTIMATE FIRED"
            elif self._ult_latched:
                out.message = "SEPARATE YOUR HANDS TO RE-ARM"
            else:
                out.message = "ULTIMATE COOLING DOWN"
            # a fresh pose must not leak into charge/shield while the two-hand pose is active
            self.pose = self._debounce(raw_pose, dt)
            self._open_t = self._fist_t = 0.0
            self._shield_on = False
            out.pose = self.pose
            out.charge_held = False
            out.shield_held = False
            out.aim_point = (cx, cy)
            self._last_aim, self._last_aim_age = (cx, cy), 0.0
            return out
        # two-hand pose not active
        self._ult_gap += dt
        if self._ult_gap > cfg.ultimate_loss_grace:
            self._ult_t = 0.0
            if self._ult_cd <= 0:
                self._ult_latched = False
        if len(usable) == 1:
            out.message = ""

        # ---- single-hand pose logic -------------------------------------
        prev_pose = self.pose
        pose = self._debounce(raw_pose, dt)
        out.pose = pose
        self._last_aim, self._last_aim_age = (cx, cy), 0.0
        out.aim_point = (cx, cy)

        if pose == HandPose.OPEN:
            self._open_t += dt
            self._since_open = 0.0
        else:
            self._open_t = 0.0
            self._since_open += dt
        if pose == HandPose.FIST:
            if prev_pose == HandPose.OPEN or (self._since_open < cfg.fire_fist_window and self._fist_t == 0.0):
                self._fist_blocked = True            # this fist is the "fire" gesture
            self._fist_t += dt
        else:
            self._fist_t = 0.0
            self._fist_blocked = False

        # ki blast: open palm charges; any other debounced pose releases (fires)
        if self._charging:
            if pose == HandPose.OPEN:
                self._other_t = 0.0
            elif pose == HandPose.OTHER:
                self._other_t += dt
                if self._other_t >= cfg.other_grace:
                    self._charging = False
            else:
                self._charging = False
        elif pose == HandPose.OPEN and self._open_t >= cfg.charge_start_delay:
            self._charging = True
            self._other_t = 0.0
        out.charge_held = self._charging

        # shield: sustained fist that is not a fire-fist
        if pose == HandPose.FIST:
            self._shield_gap = 0.0
            if not self._fist_blocked and self._fist_t >= cfg.shield_hold:
                self._shield_on = True
        elif self._shield_on:
            self._shield_gap += dt
            if self._shield_gap > cfg.shield_release_grace:
                self._shield_on = False
        out.shield_held = self._shield_on and not self._charging

        if out.charge_held:
            out.action = "CHARGING KI - CLOSE HAND TO FIRE"
        elif out.shield_held:
            out.action = "SHIELD UP"
        elif pose == HandPose.FIST and not self._fist_blocked and self._fist_t > 0:
            out.action = "RAISING SHIELD..."
        elif len(usable) == 1:
            out.message = "TWO HANDS TOGETHER = ULTIMATE"
        return out