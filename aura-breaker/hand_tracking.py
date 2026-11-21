"""MediaPipe Hand Landmarker (Tasks API) wrapper.

Runs on the frames the face tracker already receives (see ``TrackingPipeline`` in
``face_tracking.py``), so no second camera is ever opened.

Model asset: ``assets/models/hand_landmarker.task`` (~7.5 MB, Google, Apache-2.0).
It is downloaded automatically on first use when the network is available, or run:

    python hand_tracking.py --download
"""
from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np

import config as C
from hand_controls import HandObs

log = logging.getLogger("aura.hands")

HAND_MODEL_FILENAME = "hand_landmarker.task"
HAND_MODEL_PATH = C.MODEL_DIR / HAND_MODEL_FILENAME
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_MIN_BYTES = 1_000_000


class HandTrackerUnavailable(RuntimeError):
    """Raised when MediaPipe or the hand model asset cannot be used."""


def ensure_hand_model(path: Path = HAND_MODEL_PATH, timeout: float = 20.0) -> Path:
    """Return the model path, downloading it first if it is missing or truncated."""
    if path.exists() and path.stat().st_size >= HAND_MODEL_MIN_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        log.info("Downloading hand model to %s ...", path)
        with urllib.request.urlopen(HAND_MODEL_URL, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        if tmp.stat().st_size < HAND_MODEL_MIN_BYTES:
            raise OSError("downloaded file is too small")
        tmp.replace(path)
    except Exception as exc:  # network errors, disk errors, truncated download
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise HandTrackerUnavailable(
            f"Hand model missing at {path} and download failed ({exc}). "
            f"Download it manually from {HAND_MODEL_URL}"
        ) from exc
    return path


class HandTracker:
    """Thin wrapper around ``vision.HandLandmarker`` in VIDEO mode."""

    def __init__(self, model_path: Path = HAND_MODEL_PATH, num_hands: int = 2, auto_download: bool = True) -> None:
        if auto_download:
            model_path = ensure_hand_model(model_path)
        elif not model_path.exists() or model_path.stat().st_size < HAND_MODEL_MIN_BYTES:
            raise HandTrackerUnavailable(f"Hand model missing or incomplete at {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise HandTrackerUnavailable(f"MediaPipe is not installed: {exc}") from exc
        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # MediaPipe raises generic RuntimeError/ValueError
            raise HandTrackerUnavailable(f"Could not create HandLandmarker: {exc}") from exc
        self._mp = mp
        self._last_ts = 0

    def process(self, rgb: np.ndarray, ts_ms: int) -> list[HandObs]:
        ts = max(int(ts_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        hands: list[HandObs] = []
        for i, pts in enumerate(result.hand_landmarks or []):
            lm = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)
            score, label = 1.0, ""
            try:
                cat = result.handedness[i][0]
                score, label = float(cat.score), str(cat.category_name)
            except (IndexError, AttributeError):
                pass
            hands.append(HandObs(lm, score, label))
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception as exc:
            log.debug("Hand landmarker close: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--download" in sys.argv:
        try:
            print("OK:", ensure_hand_model())
        except HandTrackerUnavailable as exc:
            print("FAILED:", exc)
            sys.exit(1)
    else:
        print(__doc__)