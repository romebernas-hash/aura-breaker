"""Central configuration for AURA BREAKER.

All balancing constants live here so that no module contains magic numbers.
User-adjustable values are stored in :class:`Settings` and persisted to
``settings.json``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("aura")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_DIR = ASSETS / "models"
IMAGE_DIR = ASSETS / "images"
EFFECT_DIR = ASSETS / "effects"
SOUND_DIR = ASSETS / "sounds"
FONT_DIR = ASSETS / "fonts"
SETTINGS_PATH = ROOT / "settings.json"

MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_MIN_BYTES = 1_000_000

HAIR_PATH = IMAGE_DIR / "hair_gold.png"
ENEMY_SPRITE_PATH = IMAGE_DIR / "enemy.png"  # optional user-supplied sprite

# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
TITLE = "AURA BREAKER: SAIYAN AWAKENING"
TAGLINE = "YOUR FACE. YOUR POWER. YOUR BATTLE."
LOGICAL_W, LOGICAL_H = 1280, 720
TARGET_FPS = 60
MAX_DT = 1 / 20  # clamp frame delta so a hitch never teleports the simulation

ENEMY_POS = (1010, 380)
ENEMY_RADIUS = 110
PLAYER_DEFAULT_POS = (330, 330)

# --------------------------------------------------------------------------
# Battle balance
# --------------------------------------------------------------------------
PLAYER_MAX_HP = 100.0
PLAYER_MAX_ENERGY = 100.0
ENERGY_REGEN = 8.0            # per second while idle
TRAINING_ENERGY_REGEN = 30.0
BASE_POWER = 9000             # power level at full energy, no multipliers
UPGRADE_POWER_BONUS = 0.08    # +8% per round cleared

# Transformation
TRANSFORM_COST = 30.0
TRANSFORM_ANIM = 2.6
TRANSFORM_DURATION = 22.0
TRANSFORM_DRAIN = 3.0          # energy / s while transformed (no regen)
TRANSFORM_MULT = 1.5
TRANSFORM_COOLDOWN = 14.0

# Energy blaster
BLASTER_MIN_ENERGY = 10.0     # must exceed COST_MIN + MIN_CHARGE*span
BLASTER_MIN_CHARGE = 0.10
BLASTER_CHARGE_TIME = 1.6      # seconds to reach full charge
BLASTER_COST_MIN = 8.0
BLASTER_COST_MAX = 25.0
BLASTER_DMG_MIN = 8.0
BLASTER_DMG_MAX = 30.0
BLASTER_COOLDOWN = 0.6
BEAM_TRAVEL_TIME = 0.20        # beam head reaches the target after this long
BEAM_DURATION = 0.55
AIM_TOLERANCE = 0.40           # |aim - enemy_aim| allowed for a hit (aim units)
MAX_AIM_DEG = 24.0             # aim=1 rotates the beam by this many degrees
STAGGER_DAMAGE = 20.0          # hits at least this strong interrupt telegraphs

# Scanner
SCAN_COST = 5.0
SCAN_DURATION = 5.0
SCAN_COOLDOWN = 3.0

# Rage burst
RAGE_COST = 25.0
RAGE_ANIM = 0.9
RAGE_DURATION = 8.0
RAGE_MULT = 1.6
RAGE_COOLDOWN = 18.0

# Shield
SHIELD_MIN_ENERGY = 10.0
SHIELD_DRAIN = 14.0
SHIELD_MAX_STRENGTH = 60.0
SHIELD_REDUCTION = 0.70
SHIELD_REGEN = 10.0
SHIELD_BREAK_LOCK = 3.0
SHIELD_MIN_STRENGTH = 12.0   # strength needed to (re)raise the shield

# Ultimate
ULTIMATE_COST = 90.0
ULTIMATE_CHARGE_TIME = 1.8
ULTIMATE_TOTAL_TIME = 3.2
ULTIMATE_DAMAGE = 70.0
ULTIMATE_COOLDOWN = 45.0

# Cinematic invulnerability during locked animations
INVULN_STATES = ("TRANSFORMING", "RAGE_BURST", "ULTIMATE")

# Enemy
ENEMY_BASE_HP = 120.0
ENEMY_HP_SCALE = 1.25
ENEMY_BASE_INTERVAL = 3.2
ENEMY_INTERVAL_SCALE = 0.88
ENEMY_MIN_INTERVAL = 1.5
ENEMY_BASE_TELEGRAPH = 1.1
ENEMY_TELEGRAPH_SCALE = 0.93
ENEMY_MIN_TELEGRAPH = 0.65
ENEMY_BASE_DAMAGE = 14.0
ENEMY_DAMAGE_SCALE = 1.15
ENEMY_PROJECTILE_TIME = 0.55
ENEMY_ATTACK_ANIM = 0.30
ENEMY_RECOVER_TIME = 0.7
ENEMY_HIT_TIME = 0.35
ENEMY_BASE_POWER = 12000
ROUND_HEAL = 30.0

# Score
SCORE_PER_DAMAGE = 10
COMBO_WINDOW = 4.0
COMBO_MAX_MULT = 3.0
COMBO_STEP = 0.1
VICTORY_BONUS = 1000

# --------------------------------------------------------------------------
# Face control defaults
# --------------------------------------------------------------------------
GESTURE_ON = 0.65
GESTURE_OFF = 0.40
GESTURE_NEUTRAL = 0.25
GESTURE_SMOOTH_TAU = 0.05      # seconds
GESTURE_RELEASE_COOLDOWN = 0.30
FACE_LOSS_GRACE = 0.5
SCAN_HOLD = 0.25
ULTIMATE_HOLD = 0.60
AIM_ROLL_RANGE_DEG = 18.0
AIM_DEADZONE_DEG = 2.5

# Default neutral baselines (used only until auto-baseline completes)
DEFAULT_BASELINE = {"mouth_open": 0.02, "brow_raise": 0.075, "lip_width": 0.40, "roll_deg": 0.0}
# Default deltas from baseline that count as a "full" gesture (score = 1.0)
DEFAULT_DELTAS = {"MOUTH_OPEN": 0.07, "BROW_RAISE": 0.018, "BROW_LOWER": -0.014, "PURSE": -0.07}
# Minimum separation between neutral and gesture sample to accept calibration
MIN_SEPARATION = {"MOUTH_OPEN": 0.03, "BROW_RAISE": 0.006, "BROW_LOWER": 0.005, "PURSE": 0.02}
AUTO_BASELINE_SECONDS = 1.0

CALIB_PREP_TIME = 2.0
CALIB_RECORD_TIME = 2.0
CALIB_NEUTRAL_TIME = 2.0

# --------------------------------------------------------------------------
# Effects quality
# --------------------------------------------------------------------------
QUALITY = {
    "low":      {"particles": 160, "bolts": 2, "tongues": 8,  "bloom": 0, "spawn": 0.4},
    "standard": {"particles": 420, "bolts": 4, "tongues": 12, "bloom": 1, "spawn": 0.8},
    "high":     {"particles": 800, "bolts": 6, "tongues": 18, "bloom": 2, "spawn": 1.0},
}
QUALITY_ORDER = ["low", "standard", "high"]

VOLUME_CHANNELS = ["music", "transform", "charge", "beam", "impact", "shield", "result"]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    fullscreen: bool = False
    quality: str = "standard"
    muted: bool = False
    master_volume: float = 0.8
    volumes: dict[str, float] = field(
        default_factory=lambda: {c: (0.5 if c == "music" else 0.9) for c in VOLUME_CHANNELS}
    )
    camera_index: int = 0
    mirror: bool = True
    show_landmarks: bool = False
    face_controls: bool = True
    invert_tilt: bool = False
    sensitivity: float = 1.0
    hold_ms: int = 120
    brow_action: str = "auto"      # auto | transform | rage
    calibration: dict[str, Any] | None = None
    first_run_done: bool = False

    # -- persistence -------------------------------------------------------
    def clamp(self) -> "Settings":
        self.quality = self.quality if self.quality in QUALITY else "standard"
        self.master_volume = min(1.0, max(0.0, float(self.master_volume)))
        for c in VOLUME_CHANNELS:
            self.volumes[c] = min(1.0, max(0.0, float(self.volumes.get(c, 0.8))))
        self.sensitivity = min(2.0, max(0.4, float(self.sensitivity)))
        self.hold_ms = int(min(400, max(0, int(self.hold_ms))))
        self.camera_index = int(min(9, max(0, int(self.camera_index))))
        if self.brow_action not in ("auto", "transform", "rage"):
            self.brow_action = "auto"
        return self

    def save(self, path: Path = SETTINGS_PATH) -> None:
        try:
            path.write_text(json.dumps(asdict(self), indent=2))
        except OSError as exc:
            log.error("Could not save settings to %s: %s", path, exc)

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for key, value in data.items():
                    if key == "volumes" and isinstance(value, dict):
                        s.volumes.update({k: v for k, v in value.items() if k in VOLUME_CHANNELS})
                    elif hasattr(s, key):
                        setattr(s, key, value)
            except (OSError, ValueError, TypeError) as exc:
                log.error("Ignoring unreadable settings file %s: %s", path, exc)
        return s.clamp()
