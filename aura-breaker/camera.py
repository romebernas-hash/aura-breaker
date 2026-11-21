"""Camera access (OpenCV) and webcam-frame -> logical-screen mapping."""
from __future__ import annotations

import logging
import sys
import time

import config as C

log = logging.getLogger("aura.camera")


class FrameMapper:
    """Maps normalized landmark coordinates of a webcam frame onto the logical
    screen using a centered "cover" crop (the frame fills the screen, extra
    pixels are cropped). Mirroring is handled here so the on-screen result
    behaves like a mirror."""

    def __init__(self, fw: int, fh: int, lw: int = C.LOGICAL_W, lh: int = C.LOGICAL_H, mirror: bool = True) -> None:
        self.fw, self.fh, self.lw, self.lh, self.mirror = max(1, fw), max(1, fh), lw, lh, mirror
        target = lw / lh
        if self.fw / self.fh > target:
            self.crop_h = float(self.fh)
            self.crop_w = self.fh * target
        else:
            self.crop_w = float(self.fw)
            self.crop_h = self.fw / target
        self.crop_x = (self.fw - self.crop_w) / 2
        self.crop_y = (self.fh - self.crop_h) / 2

    @property
    def crop_rect(self) -> tuple[int, int, int, int]:
        return int(self.crop_x), int(self.crop_y), int(self.crop_w), int(self.crop_h)

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        """``nx, ny`` are normalized coordinates in the *raw* (unmirrored) frame."""
        if self.mirror:
            nx = 1.0 - nx
        px = nx * self.fw
        py = ny * self.fh
        return (
            (px - self.crop_x) / self.crop_w * self.lw,
            (py - self.crop_y) / self.crop_h * self.lh,
        )


class Camera:
    """Thin wrapper around cv2.VideoCapture with macOS-friendly backend choice."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap = None
        self.error = ""

    def open(self, warmup_seconds: float = 3.0) -> bool:
        try:
            import cv2
        except ImportError:
            self.error = "OpenCV (cv2) is not installed."
            return False
        backends = []
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.append(cv2.CAP_ANY)
        for backend in backends:
            try:
                cap = cv2.VideoCapture(self.index, backend)
            except Exception as exc:  # OpenCV can raise on odd backends
                log.warning("VideoCapture(%s, %s) raised: %s", self.index, backend, exc)
                continue
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Some cameras need a moment before delivering the first frame.
            deadline = time.time() + warmup_seconds
            while time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    log.info("Camera %d opened (%dx%d)", self.index, frame.shape[1], frame.shape[0])
                    return True
                time.sleep(0.05)
            cap.release()
        self.error = (
            f"Could not open camera {self.index}. On macOS, allow camera access for your terminal / "
            "VS Code in System Settings > Privacy & Security > Camera, then restart it."
        )
        return False

    def read(self):
        if self.cap is None:
            return False, None
        try:
            return self.cap.read()
        except Exception as exc:
            log.error("Camera read failed: %s", exc)
            return False, None

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                log.error("Camera release failed: %s", exc)
            self.cap = None
