"""MediaPipe Face Landmarker (Tasks API) + Hand Landmarker wrapper and background tracking thread.

Only the Tasks API is used (``mediapipe.tasks.python.vision``); the legacy
``mp.solutions`` API is intentionally not mixed in.

One camera, one worker thread: every frame is run through the face tracker and
(optionally) the hand tracker, so the webcam is only ever opened once.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import config as C
from camera import Camera, FrameMapper
from face_controls import FaceMeasurements, measure_landmarks
from hand_controls import HandObs
from hand_tracking import HandTracker, HandTrackerUnavailable

log = logging.getLogger("aura.tracking")


class TrackerUnavailable(RuntimeError):
    """Raised when MediaPipe or the model asset cannot be used."""


@dataclass
class TrackingResult:
    frame: np.ndarray | None            # RGB, already mirrored if requested
    landmarks: np.ndarray | None        # (478, 3) normalized, raw-frame coordinates
    measurements: FaceMeasurements | None
    seq: int = 0
    size: tuple[int, int] = (0, 0)      # raw frame (w, h)
    mirror: bool = True
    hands: list[HandObs] | None = None  # detected hands (raw-frame normalized landmarks)


class FaceTracker:
    def __init__(self, model_path: Path = C.MODEL_PATH) -> None:
        if not model_path.exists() or model_path.stat().st_size < C.MODEL_MIN_BYTES:
            raise TrackerUnavailable(
                f"Face model missing or incomplete at {model_path}. Run: python scripts/setup_assets.py"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise TrackerUnavailable(f"MediaPipe is not installed: {exc}") from exc
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # MediaPipe raises generic RuntimeError/ValueError
            raise TrackerUnavailable(f"Could not create FaceLandmarker: {exc}") from exc
        self._mp = mp
        self._last_ts = 0

    def process(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        ts = max(int(ts_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        pts = result.face_landmarks[0]
        return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception as exc:
            log.debug("Landmarker close: %s", exc)


class TrackingPipeline:
    """Camera capture + face/hand inference on a worker thread.

    The render loop only ever calls :meth:`latest`, so slow inference never
    blocks drawing or input.
    """

    def __init__(self, camera_index: int = 0, mirror: bool = True, use_tracker: bool = True,
                 use_hands: bool = True) -> None:
        self.camera_index = camera_index
        self.mirror = mirror
        self.use_tracker = use_tracker
        self.use_hands = use_hands
        self.status = "starting"          # starting | running | no_camera | camera_lost
        self.tracker_status = "starting"  # starting | ok | disabled | unavailable
        self.hand_status = "starting" if use_hands else "disabled"   # starting | ok | disabled | unavailable
        self.error = ""
        self.hand_error = ""
        self.fps = 0.0
        self._latest: TrackingResult | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def latest(self) -> TrackingResult | None:
        return self._latest

    @property
    def camera_ok(self) -> bool:
        return self.status == "running"

    def _run(self) -> None:
        import cv2

        cam = Camera(self.camera_index)
        if not cam.open():
            self.status, self.error = "no_camera", cam.error
            self.hand_status = "disabled"
            log.error(cam.error)
            return
        tracker: FaceTracker | None = None
        if self.use_tracker:
            try:
                tracker = FaceTracker()
                self.tracker_status = "ok"
            except TrackerUnavailable as exc:
                self.tracker_status, self.error = "unavailable", str(exc)
                log.error("Face tracking unavailable: %s", exc)
        else:
            self.tracker_status = "disabled"
        hand_tracker: HandTracker | None = None
        if self.use_hands:
            try:
                hand_tracker = HandTracker()
                self.hand_status = "ok"
            except HandTrackerUnavailable as exc:
                self.hand_status, self.hand_error = "unavailable", str(exc)
                log.error("Hand tracking unavailable (keyboard fallback stays active): %s", exc)
        self.status = "running"
        seq, fails, infer_errors, hand_errors = 0, 0, 0, 0
        t_prev = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, bgr = cam.read()
                if not ok or bgr is None:
                    fails += 1
                    if fails > 100:
                        self.status = "camera_lost"
                        self.error = "Camera stopped delivering frames."
                        break
                    time.sleep(0.01)
                    continue
                fails = 0
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                lm = meas = None
                hands: list[HandObs] | None = None
                small = rgb
                if (tracker is not None or hand_tracker is not None) and w > 640:
                    small = cv2.resize(rgb, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
                ts_ms = int(time.monotonic() * 1000)
                if tracker is not None:
                    try:
                        lm = tracker.process(small, ts_ms)
                        infer_errors = 0
                    except Exception as exc:
                        infer_errors += 1
                        log.error("Face inference error (%d): %s", infer_errors, exc)
                        if infer_errors >= 30:
                            tracker.close()
                            tracker = None
                            self.tracker_status = "unavailable"
                            self.error = f"Face tracking stopped after repeated errors: {exc}"
                    if lm is not None:
                        meas = measure_landmarks(lm, w / h)
                if hand_tracker is not None:
                    try:
                        hands = hand_tracker.process(small, ts_ms)
                        hand_errors = 0
                    except Exception as exc:
                        hand_errors += 1
                        hands = []
                        log.error("Hand inference error (%d): %s", hand_errors, exc)
                        if hand_errors >= 30:
                            hand_tracker.close()
                            hand_tracker = None
                            self.hand_status = "unavailable"
                            self.hand_error = f"Hand tracking stopped after repeated errors: {exc}"
                display = np.ascontiguousarray(cv2.flip(rgb, 1) if self.mirror else rgb)
                seq += 1
                self._latest = TrackingResult(display, lm, meas, seq, (w, h), self.mirror, hands)
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, now - t_prev))
                t_prev = now
        finally:
            cam.release()
            if tracker is not None:
                tracker.close()
            if hand_tracker is not None:
                hand_tracker.close()


class SyntheticFace:
    """Scripted face source used by ``--demo-face`` and the self-test.

    It feeds *measurements* (not landmarks) through the real gesture pipeline,
    so it exercises detection, mapping and battle integration without a webcam.
    """

    status = "running"
    tracker_status = "ok"
    hand_status = "disabled"
    error = ""
    hand_error = ""
    fps = 60.0
    camera_ok = False

    def __init__(self, script: Callable[[float], FaceMeasurements | None]) -> None:
        self.script = script
        self._t0 = time.monotonic()
        self._seq = 0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def stop(self) -> None:
        pass

    def latest(self) -> TrackingResult:
        self._seq += 1
        return TrackingResult(None, None, self.script(time.monotonic() - self._t0), self._seq)


def neutral_measurements(**overrides: float) -> FaceMeasurements:
    d = dict(C.DEFAULT_BASELINE)
    d.update(overrides)
    return FaceMeasurements(d["mouth_open"], d["brow_raise"], d["lip_width"], d["roll_deg"])


# --------------------------------------------------------------------------
# Player anchor (where effects attach)
# --------------------------------------------------------------------------
@dataclass
class PlayerAnchor:
    head: tuple[float, float]
    hairline: tuple[float, float]
    mouth: tuple[float, float]
    face_w: float
    face_h: float
    roll_deg: float          # pygame convention: positive = counter-clockwise on screen
    present: bool = False
    hand: tuple[float, float] | None = None   # firing-hand position; beams / charge orbs spawn here when set

    @classmethod
    def from_params(cls, head, face_w, face_h, roll_deg=0.0, present=False) -> "PlayerAnchor":
        hx, hy = head
        r = math.radians(roll_deg)
        up = (math.sin(r), -math.cos(r))  # screen-space "up" for a rotated head
        hairline = (hx + up[0] * face_h * 0.54, hy + up[1] * face_h * 0.54)
        mouth = (hx - up[0] * face_h * 0.22, hy - up[1] * face_h * 0.22)
        return cls((hx, hy), hairline, mouth, face_w, face_h, roll_deg, present)

    @classmethod
    def default(cls) -> "PlayerAnchor":
        return cls.from_params(C.PLAYER_DEFAULT_POS, 190.0, 250.0)

    @property
    def aura_center(self) -> tuple[float, float]:
        return (self.head[0], self.head[1] + self.face_h * 0.85)

    @property
    def aura_radii(self) -> tuple[float, float]:
        return (self.face_w * 1.3, self.face_h * 1.6)


def compute_anchor(lm: np.ndarray, mapper: FrameMapper) -> PlayerAnchor | None:
    try:
        def P(i: int) -> tuple[float, float]:
            return mapper.to_screen(float(lm[i, 0]), float(lm[i, 1]))

        fore, chin = P(10), P(152)
        cl, cr = P(234), P(454)
        top, bot = P(13), P(14)
        ea, eb = P(33), P(263)
    except (IndexError, TypeError):
        return None
    face_w = math.dist(cl, cr)
    face_h = math.dist(fore, chin)
    if face_w < 8 or face_h < 8 or not all(map(math.isfinite, (face_w, face_h))):
        return None
    if ea[0] > eb[0]:
        ea, eb = eb, ea
    roll = -math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0]))
    head = ((fore[0] + chin[0]) / 2, (fore[1] + chin[1]) / 2)
    anchor = PlayerAnchor.from_params(head, face_w, face_h, roll, present=True)
    anchor.mouth = ((top[0] + bot[0]) / 2, (top[1] + bot[1]) / 2)
    anchor.hairline = (fore[0] + (fore[0] - chin[0]) * 0.03, fore[1] + (fore[1] - chin[1]) * 0.03)
    return anchor


class AnchorSmoother:
    """Exponential smoothing of the anchor to reduce landmark jitter."""

    def __init__(self, tau: float = 0.06) -> None:
        self.tau = tau
        self.value = PlayerAnchor.default()

    def update(self, target: PlayerAnchor, dt: float) -> PlayerAnchor:
        a = 1.0 - math.exp(-max(dt, 1e-4) / self.tau)
        v = self.value

        def mix(p, q):
            return (p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a)

        self.value = PlayerAnchor(
            mix(v.head, target.head),
            mix(v.hairline, target.hairline),
            mix(v.mouth, target.mouth),
            v.face_w + (target.face_w - v.face_w) * a,
            v.face_h + (target.face_h - v.face_h) * a,
            v.roll_deg + (target.roll_deg - v.roll_deg) * a,
            target.present,
        )
        return self.value