#!/usr/bin/env python3
"""Create / download every asset AURA BREAKER needs.

* Hair overlay PNG and sounds are original and generated procedurally.
* The MediaPipe Face Landmarker model is downloaded from Google's official
  model storage (required for face tracking; the game also runs without it
  using keyboard controls).

Usage:
    python scripts/setup_assets.py              # everything missing
    python scripts/setup_assets.py --force      # regenerate generated assets
    python scripts/setup_assets.py --skip-model # do not download the model
    python scripts/setup_assets.py --model-only
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import config as C  # noqa: E402

SR = 22050


# --------------------------------------------------------------------------
# Synthesis helpers
# --------------------------------------------------------------------------
def t_axis(dur: float) -> np.ndarray:
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def env_ad(n: int, attack: float, decay_pow: float = 3.0) -> np.ndarray:
    x = np.linspace(0, 1, n)
    a = np.clip(x / max(attack, 1e-4), 0, 1)
    return a * (1 - x) ** decay_pow


def sweep(f0: float, f1: float, dur: float, power: float = 1.0) -> np.ndarray:
    t = t_axis(dur)
    k = (t / dur) ** power
    f = f0 + (f1 - f0) * k
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saw(freq, dur: float) -> np.ndarray:
    t = t_axis(dur)
    return 2 * ((t * freq) % 1.0) - 1


def noise(dur: float, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, int(SR * dur))


def lowpass(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    return np.convolve(x, np.ones(k) / k, mode="same")


def fade_edges(x: np.ndarray, ms: float = 8.0) -> np.ndarray:
    n = max(1, int(SR * ms / 1000))
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def write_wav(path: Path, x: np.ndarray, peak: float = 0.8) -> None:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x)) or 1.0
    pcm = (fade_edges(x / m * peak) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def mix(*parts: tuple[np.ndarray, float], length: int | None = None) -> np.ndarray:
    n = length or max(len(p) for p, _ in parts)
    out = np.zeros(n)
    for p, g in parts:
        out[: len(p)] += p[:n] * g
    return out


def note(freq: float, dur: float, decay: float = 3.0, harm: bool = True) -> np.ndarray:
    t = t_axis(dur)
    x = np.sin(2 * np.pi * freq * t)
    if harm:
        x += 0.4 * np.sin(2 * np.pi * freq * 2 * t) + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    return x * env_ad(len(t), 0.01, decay)


# --------------------------------------------------------------------------
# Sounds
# --------------------------------------------------------------------------
def make_sounds(force: bool) -> None:
    C.SOUND_DIR.mkdir(parents=True, exist_ok=True)

    def out(name: str) -> Path:
        return C.SOUND_DIR / name

    def want(name: str) -> bool:
        return force or not out(name).exists()

    if want("charge.wav"):  # seamless-ish loop: periodic LFO
        d = 1.2
        t = t_axis(d)
        f = 220 + 90 * np.sin(2 * np.pi * t / d) + 40 * np.sin(2 * np.pi * 3 * t / d)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.7 + 0.3 * np.sin(2 * np.pi * 8 * t))
        x += 0.25 * lowpass(noise(d, 2), 6)
        write_wav(out("charge.wav"), x, 0.55)
    if want("beam.wav"):
        d = 0.9
        n = int(SR * d)
        x = mix((sweep(900, 90, d, 0.5), 0.8), (lowpass(noise(d, 3), 4) * np.linspace(1, 0.1, n), 0.6),
                (saw(70, d), 0.4))
        write_wav(out("beam.wav"), x * env_ad(n, 0.01, 1.6))
    if want("impact.wav"):
        d = 0.55
        n = int(SR * d)
        x = mix((sweep(110, 38, d, 0.6) * env_ad(n, 0.002, 2.5), 1.0), (lowpass(noise(d, 4), 12) * env_ad(n, 0.001, 4), 0.8))
        write_wav(out("impact.wav"), x)
    if want("transform.wav"):
        d = 2.6
        n = int(SR * d)
        rise = sweep(70, 1100, d, 2.0) * np.linspace(0.1, 1, n) ** 1.5
        swell = lowpass(noise(d, 5), 8) * np.linspace(0, 1, n) ** 2
        burst = np.zeros(n)
        b = int(SR * 0.9)
        burst[-b:] = (lowpass(noise(0.9, 6), 6) * env_ad(b, 0.01, 2.0)) + sweep(200, 60, 0.9) * env_ad(b, 0.01, 2.0)
        write_wav(out("transform.wav"), mix((rise, 0.7), (swell, 0.6), (burst, 1.0), (saw(55, d) * np.linspace(0, 1, n), 0.3)))
    if want("rage.wav"):
        d = 1.0
        n = int(SR * d)
        x = mix((saw(65, d) * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t_axis(d))), 0.6), (saw(98, d), 0.4),
                (lowpass(noise(d, 7), 5) * np.linspace(0.2, 1, n), 0.5))
        write_wav(out("rage.wav"), x * env_ad(n, 0.05, 0.8))
    if want("ultimate.wav"):
        rise = sweep(60, 1500, 1.8, 2.4) * np.linspace(0.1, 1, int(SR * 1.8)) ** 1.5
        bd = 1.4
        bn = int(SR * bd)
        blast = mix((sweep(300, 40, bd, 0.5), 1.0), (lowpass(noise(bd, 8), 6), 0.9), length=bn) * env_ad(bn, 0.005, 1.8)
        x = np.concatenate([rise, blast])
        write_wav(out("ultimate.wav"), x)
    if want("shield.wav"):
        d = 0.6
        t = t_axis(d)
        x = (np.sin(2 * np.pi * 880 * t) + 0.7 * np.sin(2 * np.pi * 1320 * t) + 0.4 * np.sin(2 * np.pi * 1760 * t))
        x *= (0.6 + 0.4 * np.sin(2 * np.pi * 18 * t)) * env_ad(len(t), 0.02, 2.0)
        write_wav(out("shield.wav"), x, 0.6)
    if want("scan.wav"):
        x = np.concatenate([note(660, 0.14, 2.5, False), note(990, 0.14, 2.5, False), note(1320, 0.2, 3.0, False)])
        write_wav(out("scan.wav"), x, 0.5)
    if want("ui.wav"):
        write_wav(out("ui.wav"), note(740, 0.07, 4.0, False), 0.4)
    if want("victory.wav"):
        notes = [261.6, 329.6, 392.0, 523.3]
        seg = [np.concatenate([np.zeros(int(SR * 0.16 * i)), note(f, 0.9, 2.2)]) for i, f in enumerate(notes)]
        chord = mix(*[(note(f, 1.4, 2.0), 0.6) for f in (261.6, 329.6, 392.0, 523.3)])
        x = mix(*[(s, 1.0) for s in seg], (np.concatenate([np.zeros(int(SR * 0.64)), chord]), 0.9))
        write_wav(out("victory.wav"), x, 0.7)
    if want("defeat.wav"):
        parts = [note(f, 0.6, 2.5) for f in (440.0, 349.2, 261.6, 196.0)]
        x = np.concatenate(parts)
        write_wav(out("defeat.wav"), x, 0.6)
    if want("music.wav"):
        bpm = 120
        beat = 60 / bpm
        beats = 16
        total = int(SR * beat * beats)
        track = np.zeros(total)
        bass = [55.0, 55.0, 43.65, 43.65, 49.0, 49.0, 41.2, 41.2]  # A1 F1 G1 E1 (per 2 beats)
        for i, f in enumerate(bass):
            s = int(SR * beat * 2 * i)
            seg = note(f, beat * 2, 1.6) + 0.3 * saw(f, beat * 2) * env_ad(int(SR * beat * 2), 0.01, 2.0)
            track[s : s + len(seg)] += seg[: total - s] * 0.9
        arp = [220.0, 261.6, 329.6, 261.6, 174.6, 220.0, 261.6, 220.0, 196.0, 246.9, 293.7, 246.9, 164.8, 207.7, 246.9, 207.7]
        step = beat / 2
        for i in range(beats * 2):
            f = arp[i % len(arp)] * 2
            s = int(SR * step * i)
            seg = note(f, step * 1.6, 3.5)
            track[s : s + len(seg)] += seg[: total - s] * 0.35
        for i in range(beats):  # soft pulse
            s = int(SR * beat * i)
            seg = sweep(120, 45, 0.18, 0.5) * env_ad(int(SR * 0.18), 0.002, 3.0)
            track[s : s + len(seg)] += seg[: total - s] * (0.7 if i % 2 == 0 else 0.35)
        write_wav(out("music.wav"), track, 0.45)
    print(f"[ok] sounds in {C.SOUND_DIR}")


# --------------------------------------------------------------------------
# Hair
# --------------------------------------------------------------------------
def make_hair(force: bool) -> None:
    import pygame

    from effects import make_hair_surface

    C.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if C.HAIR_PATH.exists() and not force:
        print(f"[ok] hair asset exists: {C.HAIR_PATH}")
        return
    pygame.image.save(make_hair_surface(), str(C.HAIR_PATH))
    print(f"[ok] generated hair overlay: {C.HAIR_PATH}")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def model_ok() -> bool:
    return C.MODEL_PATH.exists() and C.MODEL_PATH.stat().st_size >= C.MODEL_MIN_BYTES


def download_model(force: bool = False) -> bool:
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_ok() and not force:
        print(f"[ok] face model present: {C.MODEL_PATH} ({C.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
        return True
    tmp = C.MODEL_PATH.with_suffix(".download")
    print(f"[..] downloading face model from {C.MODEL_URL}")
    try:
        with urllib.request.urlopen(C.MODEL_URL, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(C.MODEL_PATH)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"[!!] model download failed: {exc}")
        print("     Download it manually and save it as:")
        print(f"       {C.MODEL_PATH}")
        print(f"     curl -L -o '{C.MODEL_PATH}' '{C.MODEL_URL}'")
        return False
    if not model_ok():
        print(f"[!!] downloaded file is too small ({C.MODEL_PATH.stat().st_size} bytes); delete it and retry.")
        return False
    if not zipfile.is_zipfile(C.MODEL_PATH):
        print("[warn] model file is not a zip bundle as expected; MediaPipe may reject it.")
    print(f"[ok] face model saved: {C.MODEL_PATH} ({C.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    return True


# --------------------------------------------------------------------------
def ensure_generated_assets() -> None:
    """Called by main.py at startup: quietly create missing generated assets."""
    if not C.HAIR_PATH.exists():
        make_hair(False)
    if not (C.SOUND_DIR / "music.wav").exists() or not (C.SOUND_DIR / "impact.wav").exists():
        make_sounds(False)
    for d in (C.EFFECT_DIR, C.FONT_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="regenerate generated assets / re-download model")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--model-only", action="store_true")
    args = ap.parse_args()
    for d in (C.MODEL_DIR, C.IMAGE_DIR, C.EFFECT_DIR, C.SOUND_DIR, C.FONT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ok = True
    if not args.model_only:
        make_hair(args.force)
        make_sounds(args.force)
    if not args.skip_model:
        ok = download_model(args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
