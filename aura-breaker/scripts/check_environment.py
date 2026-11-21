#!/usr/bin/env python3
"""Verify Python, dependencies, assets and (optionally) the webcam.

    python scripts/check_environment.py
    python scripts/check_environment.py --camera     # also grab one webcam frame
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

problems: list[str] = []
warnings: list[str] = []


def ok(msg): print(f"[ok]   {msg}")
def warn(msg): print(f"[warn] {msg}"); warnings.append(msg)
def bad(msg): print(f"[FAIL] {msg}"); problems.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="try to open the webcam and read a frame")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    v = sys.version_info
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if v < (3, 9):
        bad("Python 3.9+ required (3.11 or 3.12 recommended).")
    elif v >= (3, 14):
        warn("Python 3.14+: MediaPipe wheels may not exist yet. Use Python 3.11/3.12 if installation fails.")
    else:
        ok("Python version is supported")
    if sys.prefix == sys.base_prefix:
        warn("Not running inside a virtual environment (.venv). Recommended: source .venv/bin/activate")

    for mod in ("numpy", "cv2", "pygame", "mediapipe"):
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            bad(f"{mod} import failed: {exc}")
    try:
        from mediapipe.tasks.python import vision
        assert hasattr(vision, "FaceLandmarker")
        ok("MediaPipe Tasks API (FaceLandmarker) available")
    except Exception as exc:
        bad(f"MediaPipe Tasks FaceLandmarker unavailable: {exc}")

    import config as C
    if C.MODEL_PATH.exists() and C.MODEL_PATH.stat().st_size >= C.MODEL_MIN_BYTES:
        extra = "" if zipfile.is_zipfile(C.MODEL_PATH) else " (not a zip bundle - may be corrupt)"
        ok(f"face model: {C.MODEL_PATH.name} {C.MODEL_PATH.stat().st_size / 1e6:.1f} MB{extra}")
    else:
        bad(f"face model missing: run  python scripts/setup_assets.py   (needed for face tracking)")
    for p in (C.HAIR_PATH, C.SOUND_DIR / "music.wav", C.SOUND_DIR / "impact.wav"):
        (ok if p.exists() else warn)(f"asset {p.relative_to(ROOT)} {'present' if p.exists() else 'missing (auto-generated at launch)'}")

    if args.camera:
        try:
            from camera import Camera
            cam = Camera(args.index)
            if cam.open():
                good, frame = cam.read()
                ok(f"camera {args.index}: frame {frame.shape[1]}x{frame.shape[0]}" if good else "camera opened but gave no frame")
                cam.release()
            else:
                bad(cam.error)
        except Exception as exc:
            bad(f"camera test failed: {exc}")

    print()
    if problems:
        print(f"{len(problems)} problem(s). The game can still run with keyboard controls "
              "(python main.py --no-camera) if only the face model/camera are affected.")
        return 1
    print("Environment looks good." + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
