"""Original sound pack for AURA BREAKER, synthesized with numpy (no samples, nothing from any anime or game).

Anime-style energy audio is built from layers: a low rumble, a rising tone, electric crackle and a
bright shimmer on top. Every sound below is a stack of those layers.

    python sound_design.py            # regenerate every file into assets/sounds/
    ensure_sound_pack()               # called by main.py; generates once, then only if the pack version changes

Format: 22050 Hz, mono, 16-bit WAV (matches the mixer settings in audio_manager.py).
Loops (charge, kicharge, music) are built from whole cycles so they repeat without a click.
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("aura.sound")

SR = 22050
PACK_VERSION = "v2-original-energy"
MARKER = ".sound_pack"


# --------------------------------------------------------------------------
# DSP helpers
# --------------------------------------------------------------------------
def T(d: float) -> np.ndarray:
    return np.arange(int(round(SR * d)), dtype=np.float64) / SR


def _phase(freq, n):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    return 2 * np.pi * np.cumsum(f) / SR


def sine(freq, n):
    return np.sin(_phase(freq, n))


def saw(freq, n):
    p = _phase(freq, n) / (2 * np.pi)
    return 2 * (p - np.floor(p + 0.5))


def square(freq, n):
    return np.sign(np.sin(_phase(freq, n)) + 1e-9)


def noise(n, rng):
    return rng.uniform(-1, 1, n)


def lp(x, fc):
    a = 1 - np.exp(-2 * np.pi * min(fc, SR * 0.45) / SR)
    k = int(min(len(x), max(8, 7 / a)))
    h = a * (1 - a) ** np.arange(k)
    return np.convolve(x, h)[: len(x)]


def hp(x, fc):
    return x - lp(x, fc)


def bp(x, lo, hi):
    return lp(hp(x, lo), hi)


def lp_sweep(x, f0, f1, steps=7):
    cuts = np.geomspace(f0, f1, steps)
    ys = [lp(x, c) for c in cuts]
    pos = np.linspace(0, steps - 1, len(x))
    out = np.zeros_like(x)
    for i, y in enumerate(ys):
        out += y * np.clip(1 - np.abs(pos - i), 0, 1)
    return out


def decay(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def adsr(n, a, d, s, r):
    an, dn, rn = int(a * SR), int(d * SR), int(r * SR)
    sn = max(0, n - an - dn - rn)
    env = np.concatenate([np.linspace(0, 1, an, endpoint=False), np.linspace(1, s, dn, endpoint=False),
                          np.full(sn, s), np.linspace(s, 0, rn)])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def ramp(n, a=0.0, b=1.0, curve=1.0):
    return a + (b - a) * np.linspace(0, 1, n) ** curve


def fade_io(x, fi=0.005, fo=0.02):
    x = x.copy()
    a, b = int(fi * SR), int(fo * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def midi(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def add(buf, x, at, gain=1.0, wrap=False):
    i = int(round(at * SR))
    if i >= len(buf):
        return
    end = i + len(x)
    if end <= len(buf):
        buf[i:end] += x * gain
    elif wrap:
        first = len(buf) - i
        buf[i:] += x[:first] * gain
        rest = x[first:][: len(buf)]
        buf[: len(rest)] += rest * gain
    else:
        buf[i:] += x[: len(buf) - i] * gain


def echo(x, delay, fb=0.35, taps=3):
    out = x.copy()
    for k in range(1, taps + 1):
        out += np.roll(x, int(delay * SR * k)) * (fb ** k)
    return out


def finish(x, peak=0.9, drive=1.0):
    x = np.tanh(x * drive)
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float64)


def crackles(n, rng, count, lo=0.05, hi=0.95, gain=0.6):
    out = np.zeros(n)
    for _ in range(count):
        at = rng.uniform(lo, hi) * n / SR
        ln = int(rng.uniform(0.004, 0.016) * SR)
        burst = hp(noise(ln, rng), 2500) * decay(ln, 0.004) * rng.uniform(0.4, 1.0)
        add(out, burst, min(at, n / SR - ln / SR - 0.001), gain)
    return out


# --------------------------------------------------------------------------
# Drum / instrument voices
# --------------------------------------------------------------------------
def kick():
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    f = 42 + 120 * np.exp(-t / 0.035)
    return (sine(f, n) * decay(n, 0.11) * 1.0 + np.random.default_rng(1).uniform(-1, 1, n) * decay(n, 0.004) * 0.25)


def snare(rng):
    n = int(0.22 * SR)
    return hp(noise(n, rng), 1400) * decay(n, 0.07) * 0.9 + sine(190, n) * decay(n, 0.06) * 0.5


def hat(rng, open_=False):
    n = int((0.16 if open_ else 0.05) * SR)
    return hp(noise(n, rng), 6500) * decay(n, 0.09 if open_ else 0.018)


def tom(f0):
    n = int(0.34 * SR)
    t = np.arange(n) / SR
    return sine(f0 * (1 + 0.5 * np.exp(-t / 0.05)), n) * decay(n, 0.12)


def taiko(rng):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    body = sine(46 + 55 * np.exp(-t / 0.07), n) * decay(n, 0.28)
    skin = lp(noise(n, rng), 500) * decay(n, 0.05) * 0.8
    return body + skin


def bass_note(freq, dur):
    n = int(dur * SR)
    x = lp(saw(freq, n), 520) * 0.8 + sine(freq / 2 if freq > 60 else freq, n) * 0.5
    return x * adsr(n, 0.004, 0.08, 0.55, 0.05)


def pad_note(freq, dur):
    n = int(dur * SR)
    x = sum(saw(freq * d, n) for d in (0.994, 1.0, 1.006)) / 3
    return lp(x, 1400) * adsr(n, 0.5, 0.3, 0.8, 0.6)


def pluck(freq, dur, bright=1.0):
    n = int(dur * SR)
    return lp(saw(freq, n), min(5000, freq * 5 * bright)) * decay(n, 0.16) * adsr(n, 0.002, 0.02, 1.0, 0.04)


def lead_note(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.18, 0, 1)
    x = saw(freq * vib, n) * 0.6 + square(freq * vib, n) * 0.35
    return lp(x, 3200) * adsr(n, 0.01, 0.08, 0.75, 0.1)


def bell(freq, dur=1.2):
    n = int(dur * SR)
    return (sine(freq, n) + 0.5 * sine(freq * 2.76, n) + 0.25 * sine(freq * 5.4, n)) * decay(n, 0.35)


# --------------------------------------------------------------------------
# Sound effects
# --------------------------------------------------------------------------
def snd_ui():
    n = int(0.09 * SR)
    return fade_io((sine(880, n) + 0.6 * sine(1320, n)) * decay(n, 0.03), 0.002, 0.02) * 0.8


def snd_deny():
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    return fade_io(square(np.where(t < 0.09, 240, 180), n) * 0.5 * adsr(n, 0.003, 0.02, 0.8, 0.04), 0.002, 0.02)


def snd_scan(rng):
    d = 1.1
    n = int(d * SR)
    t = np.arange(n) / SR
    sweep = sine(600 + 1800 * (t / d) ** 1.5, n) * adsr(n, 0.03, 0.2, 0.3, 0.4) * 0.5
    ping = np.zeros(n)
    for k in range(3):
        ln = int(0.25 * SR)
        add(ping, sine(1500, ln) * decay(ln, 0.07), 0.35 + 0.22 * k, 0.9 * 0.5 ** k)
    return fade_io(sweep + ping, 0.005, 0.1)


def snd_shield(rng):
    n = int(0.65 * SR)
    t = np.arange(n) / SR
    sh = sum(sine(f * (1 + 0.01 * np.sin(2 * np.pi * (6 + i) * t)), n) / (1 + 0.3 * i) for i, f in enumerate((500, 790, 1180, 1760)))
    swell = sh * adsr(n, 0.12, 0.1, 0.6, 0.3)
    boom = sine(110 * (1 + 0.6 * np.exp(-t / 0.08)), n) * decay(n, 0.2) * 0.9
    return fade_io(swell * 0.5 + boom, 0.003, 0.05)


def snd_shield_hit(rng):
    n = int(0.45 * SR)
    ring = sum(sine(f, n) * decay(n, tau) for f, tau in ((640, 0.18), (1180, 0.12), (1730, 0.09), (2310, 0.06)))
    click = hp(noise(n, rng), 3000) * decay(n, 0.01)
    thump = sine(95, n) * decay(n, 0.08) * 0.8
    return fade_io(ring * 0.35 + click * 0.6 + thump, 0.001, 0.05)


def snd_shield_break(rng):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    burst = hp(noise(n, rng), 1800) * decay(n, 0.12) * 0.8
    glass = sum(sine(f * (1 - 0.4 * t), n) * decay(n, 0.22) for f in (1900, 2400, 3100, 3700)) * 0.2
    boom = sine(80 * (1 + 0.8 * np.exp(-t / 0.1)), n) * decay(n, 0.25)
    return fade_io(burst + glass + boom, 0.001, 0.08)


def snd_impact(rng):
    n = int(0.75 * SR)
    t = np.arange(n) / SR
    crack = hp(noise(n, rng), 1200) * decay(n, 0.012) * 1.0
    blast = lp_sweep(noise(n, rng), 5000, 220) * decay(n, 0.17) * 1.1
    sub = sine(36 + 70 * np.exp(-t / 0.06), n) * decay(n, 0.28) * 1.2
    mid = sine(150 * (1 + 0.3 * np.exp(-t / 0.05)), n) * decay(n, 0.1) * 0.6
    return fade_io(crack + blast + sub + mid, 0.0005, 0.08)


def snd_hurt(rng):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    body = lp(saw(230 * (1 - 0.65 * t / 0.55), n), 1200) * decay(n, 0.16)
    hit = lp_sweep(noise(n, rng), 3000, 200) * decay(n, 0.1)
    sub = sine(60 * (1 + 0.5 * np.exp(-t / 0.05)), n) * decay(n, 0.18)
    return fade_io(body * 0.7 + hit * 0.8 + sub, 0.0008, 0.08)


def snd_warn(rng):
    d = 0.8
    n = int(d * SR)
    t = np.arange(n) / SR
    rise = sine(180 + 260 * (t / d) ** 1.6, n) * (0.55 + 0.45 * np.sin(2 * np.pi * 13 * t))
    drone = lp(saw(70, n), 400) * 0.7
    swell = lp_sweep(noise(n, rng), 400, 3000) * ramp(n, 0, 0.6)
    return fade_io((rise * 0.5 + drone + swell) * adsr(n, 0.05, 0.05, 0.9, 0.12), 0.01, 0.1)


def snd_enemy_attack(rng):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    whoosh = lp_sweep(hp(noise(n, rng), 300), 600, 4500) * adsr(n, 0.12, 0.1, 0.6, 0.2)
    zap = sine(520 * np.exp(-t / 0.2) + 90, n) * decay(n, 0.2)
    return fade_io(whoosh * 0.9 + zap * 0.7, 0.003, 0.08)


def snd_beam(rng):
    d = 1.1
    n = int(d * SR)
    t = np.arange(n) / SR
    zap = (sine(160 + 1500 * np.exp(-t / 0.09), n) * 0.7 + saw(120 + 700 * np.exp(-t / 0.09), n) * 0.35) * decay(n, 0.22)
    whoosh = lp_sweep(hp(noise(n, rng), 400), 5000, 500) * decay(n, 0.3) * 0.9
    boom = sine(48 + 110 * np.exp(-t / 0.05), n) * decay(n, 0.3) * 1.1
    shimmer = (sine(720, n) + sine(1080, n) * 0.7) * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * t)) * decay(n, 0.45) * 0.25
    crack = hp(noise(n, rng), 3000) * decay(n, 0.01) * 0.8
    return fade_io(zap + whoosh + boom + shimmer + crack, 0.0005, 0.12)


def snd_charge_start(rng):
    d = 0.95
    n = int(d * SR)
    t = np.arange(n) / SR
    rise = sine(140 * np.exp(1.9 * t / d), n) * adsr(n, 0.05, 0.1, 0.9, 0.12)
    whoosh = lp_sweep(hp(noise(n, rng), 600), 800, 5000) * ramp(n, 0.1, 1.0, 1.5)
    return fade_io(rise * 0.7 + whoosh * 0.5 + crackles(n, rng, 10, 0.4, 0.95, 0.5), 0.005, 0.1)


def snd_fizzle(rng):
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    return fade_io(sine(700 * np.exp(-3.2 * t / 0.4), n) * decay(n, 0.15) + crackles(n, rng, 6, 0.0, 0.5, 0.4), 0.002, 0.08) * 0.8


def _loop_check(x):
    return x  # loops are built from whole cycles; kept for clarity


def snd_charge_loop(rng):
    """Seamless 2 s loop: layered hum + shimmer + crackle (all frequencies are whole cycles per 2 s)."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    hum = (sine(110, n) * 0.55 + sine(220, n) * 0.35 + sine(330, n) * 0.22 + sine(55, n) * 0.4) * (0.78 + 0.22 * np.sin(2 * np.pi * 3 * t))
    shim = (sine(880, n) * 0.18 + sine(1320, n) * 0.12 + sine(1760, n) * 0.08) * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t + 1.0))
    chirp = sine(300 + 700 * ((t % 1.0)), n) * 0.12 * np.sin(np.pi * (t % 1.0))
    return hum + shim + chirp + crackles(n, rng, 16, 0.03, 0.9, 0.5)


def snd_kicharge_loop(rng):
    """Deeper, pulsing power-up loop (Ki Charge)."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    wub = 0.5 + 0.5 * np.sin(2 * np.pi * 6 * t - 1.2) ** 2
    low = (sine(55, n) * 0.7 + sine(82.5, n) * 0.5 + sine(110, n) * 0.35) * wub
    grit = np.tanh(3 * lp(saw(110, n), 700)) * 0.35 * wub
    rise = sine(240 + 1100 * (t % 1.0) ** 1.6, n) * 0.14 * np.sin(np.pi * (t % 1.0))
    air = (sine(1320, n) * 0.1 + sine(1980, n) * 0.07) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * t))
    return low + grit + rise + air + crackles(n, rng, 22, 0.03, 0.92, 0.55)


def snd_transform(rng):
    d = 3.7
    n = int(d * SR)
    t = np.arange(n) / SR
    climax = 2.55
    rumble = lp(noise(n, rng), 140) * ramp(n, 0.2, 1.4, 1.4) * 2.0
    ratio = 1 + 0.5 * np.clip(t / climax, 0, 1) ** 1.3
    choir = sum(sine(f * ratio * (1 + 0.003 * np.sin(2 * np.pi * (4.6 + i * 0.7) * t)), n) / (1 + 0.25 * i)
                for i, f in enumerate((146.8, 220.0, 293.7, 440.0, 587.3)))
    choir *= np.clip(t / 1.2, 0, 1) * (t < climax) * 0.5
    crack = crackles(n, rng, 70, 0.05, climax / d, 0.5) * ramp(n, 0.2, 1.2)
    out = rumble * (t < climax + 0.2) + choir + crack
    # burst
    bn = int(1.2 * SR)
    bt = np.arange(bn) / SR
    blast = lp_sweep(noise(bn, rng), 6000, 300) * decay(bn, 0.35) * 1.4
    boom = sine(38 + 90 * np.exp(-bt / 0.09), bn) * decay(bn, 0.55) * 1.5
    sweep = sine(3200 * np.exp(-2.6 * bt / 1.2) + 200, bn) * decay(bn, 0.5) * 0.5
    shimmer = (sine(880, bn) + sine(1320, bn) * 0.8 + sine(1760, bn) * 0.5) * decay(bn, 0.6) * 0.25
    add(out, blast + boom + sweep + shimmer, climax - 0.02, 1.0)
    return fade_io(out, 0.02, 0.2)


def snd_powerdown(rng):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    return fade_io((sine(900 * np.exp(-3.0 * t / 0.9) + 80, n) * 0.6 + lp_sweep(noise(n, rng), 3000, 200) * 0.25) * decay(n, 0.4), 0.005, 0.2)


def snd_rage(rng):
    d = 1.2
    n = int(d * SR)
    t = np.arange(n) / SR
    growl = np.tanh(3.2 * lp(saw(62 * (1 + 0.05 * np.sin(2 * np.pi * 9 * t)), n), 900)) * ramp(n, 0.3, 1.0, 1.2)
    rough = lp_sweep(noise(n, rng), 700, 2500) * ramp(n, 0.1, 0.8)
    out = growl * 0.8 + rough * 0.5
    bn = int(0.5 * SR)
    bt = np.arange(bn) / SR
    add(out, sine(44 + 100 * np.exp(-bt / 0.07), bn) * decay(bn, 0.3) * 1.6 + lp_sweep(noise(bn, rng), 4000, 250) * decay(bn, 0.15), 0.85, 1.0)
    return fade_io(out * adsr(n, 0.02, 0.05, 1.0, 0.15), 0.01, 0.15)


def snd_ultimate_charge(rng):
    """1.8 s of rising tension that peaks exactly when the final blast is released."""
    d = 1.85
    n = int(d * SR)
    t = np.arange(n) / SR
    rise = sine(120 * np.exp(2.5 * t / d), n) * ramp(n, 0.2, 1.0, 1.5)
    saws = lp(saw(55 * np.exp(1.4 * t / d), n), 1800) * ramp(n, 0.2, 0.9, 1.3)
    air = lp_sweep(hp(noise(n, rng), 700), 900, 6000) * ramp(n, 0.0, 1.0, 2.0)
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * (6 + 18 * (t / d) ** 2) * t)
    out = (rise * 0.6 + saws * 0.6 + air * 0.5) * trem
    beat, k = 0.0, 0
    while beat < d - 0.1:  # accelerating heartbeat thumps
        ln = int(0.18 * SR)
        add(out, sine(68 * (1 + 0.5 * np.exp(-np.arange(ln) / SR / 0.04)), ln) * decay(ln, 0.07), beat, 0.9)
        beat += max(0.1, 0.46 * (0.82 ** k))
        k += 1
    return fade_io(out + crackles(n, rng, 60, 0.2, 0.98, 0.5), 0.02, 0.03)


def snd_ultimate_beam(rng):
    d = 2.6
    n = int(d * SR)
    t = np.arange(n) / SR
    lift = adsr(n, 0.02, 0.2, 0.85, 1.1)
    sub = sine(32 + 60 * np.exp(-t / 0.2), n) * lift * 1.5
    roar = lp_sweep(noise(n, rng), 3500, 500) * lift * 1.0
    stack = sum(lp(saw(f * (1 + 0.004 * np.sin(2 * np.pi * 6 * t)), n), 1600) for f in (110, 165, 220, 330)) / 4 * lift * 1.1
    burst = sine(2200 * np.exp(-6 * t) + 160, n) * decay(n, 0.25) * 0.7
    shimmer = (sine(1320, n) + sine(1980, n) * 0.6) * (0.6 + 0.4 * np.sin(2 * np.pi * 8 * t)) * lift * 0.18
    return fade_io(sub + roar * 0.8 + stack + burst + shimmer + crackles(n, rng, 70, 0.02, 0.8, 0.4), 0.0005, 0.25)


def snd_ready(rng):
    n = int(0.9 * SR)
    out = np.zeros(n)
    for i, m in enumerate((76, 83, 88, 95)):
        ln = int(0.5 * SR)
        add(out, bell(midi(m), 0.5)[:ln] * 0.6, 0.07 * i)
    sweep = sine(500 * np.exp(2.0 * np.arange(n) / n), n) * adsr(n, 0.01, 0.2, 0.2, 0.4) * 0.25
    return fade_io(out + sweep, 0.002, 0.15)


def snd_victory(rng):
    d = 3.6
    n = int(d * SR)
    out = np.zeros(n)
    for i, m in enumerate((72, 76, 79, 84)):
        add(out, lead_note(midi(m), 0.4), 0.2 * i, 0.7)
    for m in (60, 64, 67, 72, 79):
        add(out, lead_note(midi(m), 2.2) * 0.55 + pad_note(midi(m - 12), 2.2) * 0.5, 0.95, 0.5)
    for at in (0.0, 0.95):
        add(out, taiko(rng), at, 0.9)
        add(out, snare(rng), at, 0.4)
    out = echo(out, 0.28, 0.3, 3)
    return fade_io(out, 0.005, 0.5)


def snd_defeat(rng):
    d = 3.0
    n = int(d * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for i, m in enumerate((57, 53, 52)):
        ln = int(1.4 * SR)
        note = lp(saw(midi(m) * (1 - 0.04 * np.clip((np.arange(ln) / SR - 0.8), 0, 1)), ln), 900) * adsr(ln, 0.05, 0.2, 0.7, 0.7)
        add(out, note, 0.55 * i, 0.7)
    out += sine(55, n) * adsr(n, 0.3, 0.3, 0.7, 1.0) * 0.5
    add(out, taiko(rng), 0.0, 0.7)
    return fade_io(echo(out, 0.35, 0.3, 2), 0.01, 0.6)


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------
def _bars_to_buf(bars, bpm):
    beat = 60.0 / bpm
    return beat, bars * 4 * beat


def music_battle():
    """16-bar original battle theme, D minor, 150 BPM. Drums + bass + strings + arpeggio + hero melody."""
    rng = np.random.default_rng(42)
    bpm = 150
    beat, total = _bars_to_buf(16, bpm)
    n = int(round(total * SR))
    buf = np.zeros(n)
    sixteenth = beat / 4
    ch = {"Dm": ([50, 53, 57], 38), "Bb": ([46, 50, 53], 34), "C": ([48, 52, 55], 36), "A": ([45, 49, 52], 33), "Gm": ([43, 46, 50], 31)}
    prog = ["Dm", "Bb", "C", "A"] * 2 + ["Dm", "Gm", "Bb", "A", "Dm", "Bb", "C", "A"]
    melody = {
        8: [(0, 2, 69), (2, 1, 74), (3, 1, 77)], 9: [(0, 2, 79), (2, 1, 77), (3, 1, 74)],
        10: [(0, 1.5, 82), (1.5, 0.5, 81), (2, 2, 77)], 11: [(0, 2, 76), (2, 1, 73), (3, 1, 76)],
        12: [(0, 2, 77), (2, 1, 81), (3, 1, 84)], 13: [(0, 2, 86), (2, 1, 84), (3, 1, 82)],
        14: [(0, 1, 84), (1, 1, 79), (2, 2, 76)], 15: [(0, 2, 81), (2, 2, 76)],
    }
    k_, s_ = kick(), snare(rng)
    h_, ho_ = hat(rng), hat(rng, True)
    toms = [tom(f) for f in (200, 160, 125, 95)]
    tk = taiko(rng)
    lead = np.zeros(n)
    for bar in range(16):
        b0 = bar * 4 * beat
        tones, root = ch[prog[bar]]
        for chord_midi in tones:   # strings pad
            add(buf, pad_note(midi(chord_midi + 12), 4 * beat), b0, 0.075, wrap=True)
        for i in range(8):         # driving eighth-note bass
            r = root + (12 if i in (3, 7) and bar % 2 else 0)
            add(buf, bass_note(midi(r), beat * 0.48), b0 + i * beat / 2, 0.33, wrap=True)
        for hit in (0, 2, 2.5):    # kick with a syncopated push
            add(buf, k_, b0 + hit * beat, 0.95, wrap=True)
        for hit in (1, 3):
            add(buf, s_, b0 + hit * beat, 0.5, wrap=True)
        for i in range(8):
            add(buf, ho_ if i % 4 == 3 else h_, b0 + i * beat / 2, 0.11 if i % 2 else 0.16, wrap=True)
        if bar % 4 == 0:
            add(buf, tk, b0, 0.55, wrap=True)
        if bar % 4 == 3:           # tom fill on the last beat
            for j in range(4):
                add(buf, toms[j], b0 + 3 * beat + j * sixteenth, 0.45, wrap=True)
        if bar < 8:                # rolling arpeggio
            pat = [0, 1, 2, 1] * 4
            for i in range(16):
                add(lead, pluck(midi(tones[pat[i]] + 24), sixteenth * 1.6, 0.8), b0 + i * sixteenth, 0.16, wrap=True)
        else:
            for i in range(8):     # quieter pulse under the melody
                add(lead, pluck(midi(tones[i % 3] + 12), sixteenth * 2.5, 0.6), b0 + i * beat / 2, 0.08, wrap=True)
        for st, du, m in melody.get(bar, []):
            add(lead, lead_note(midi(m), du * beat * 0.98), b0 + st * beat, 0.34, wrap=True)
    buf += echo(lead, beat * 0.75, 0.28, 3)
    return finish(buf, 0.82, drive=1.15)


def music_menu():
    """8-bar calm-but-heroic menu theme, D minor, 88 BPM."""
    rng = np.random.default_rng(7)
    bpm = 88
    beat, total = _bars_to_buf(8, bpm)
    n = int(round(total * SR))
    buf = np.zeros(n)
    ch = {"Dm": ([50, 53, 57], 38), "Bb": ([46, 50, 53], 34), "Gm": ([43, 46, 50], 31), "A": ([45, 49, 52], 33)}
    prog = ["Dm", "Bb", "Gm", "A"] * 2
    tk = taiko(rng)
    for bar in range(8):
        b0 = bar * 4 * beat
        tones, root = ch[prog[bar]]
        for m in tones:
            add(buf, pad_note(midi(m + 12), 4 * beat), b0, 0.16, wrap=True)
            add(buf, pad_note(midi(m), 4 * beat), b0, 0.08, wrap=True)
        add(buf, sine(midi(root), int(4 * beat * SR)) * adsr(int(4 * beat * SR), 0.4, 0.3, 0.8, 0.8), b0, 0.3, wrap=True)
        if bar % 4 == 0:
            add(buf, tk, b0, 0.7, wrap=True)
        add(buf, tk, b0 + 2 * beat, 0.25, wrap=True)
        for i in range(8):
            add(buf, bell(midi(tones[(i * 2) % 3] + 24), 1.4), b0 + i * beat / 2, 0.07, wrap=True)
    buf += echo(buf * 0.5, beat * 0.75, 0.3, 3)
    return finish(buf, 0.75, drive=1.0)


# --------------------------------------------------------------------------
# Registry / writer
# --------------------------------------------------------------------------
def _gens():
    r = lambda seed: np.random.default_rng(seed)  # noqa: E731
    return {
        "ui.wav": lambda: snd_ui(), "deny.wav": lambda: snd_deny(), "scan.wav": lambda: snd_scan(r(1)),
        "shield.wav": lambda: snd_shield(r(2)), "shield_hit.wav": lambda: snd_shield_hit(r(3)),
        "shield_break.wav": lambda: snd_shield_break(r(4)), "impact.wav": lambda: snd_impact(r(5)),
        "hurt.wav": lambda: snd_hurt(r(6)), "warn.wav": lambda: snd_warn(r(7)),
        "eattack.wav": lambda: snd_enemy_attack(r(8)), "beam.wav": lambda: snd_beam(r(9)),
        "charge_start.wav": lambda: snd_charge_start(r(10)), "fizzle.wav": lambda: snd_fizzle(r(11)),
        "charge.wav": lambda: snd_charge_loop(r(12)), "kicharge.wav": lambda: snd_kicharge_loop(r(13)),
        "transform.wav": lambda: snd_transform(r(14)), "powerdown.wav": lambda: snd_powerdown(r(15)),
        "rage.wav": lambda: snd_rage(r(16)), "ultimate.wav": lambda: snd_ultimate_charge(r(17)),
        "ultimate_beam.wav": lambda: snd_ultimate_beam(r(18)), "ready.wav": lambda: snd_ready(r(19)),
        "victory.wav": lambda: snd_victory(r(20)), "defeat.wav": lambda: snd_defeat(r(21)),
        "music.wav": music_battle, "music_menu.wav": music_menu,
    }


LOOP_FILES = {"charge.wav", "kicharge.wav", "music.wav", "music_menu.wav"}
GENERATORS = _gens()


# loudness balance (1.0 = full scale). UI blips are quiet, explosions and beams are loud, music sits underneath.
LEVEL = {
    "ui.wav": 0.30, "deny.wav": 0.22, "scan.wav": 0.55, "shield.wav": 0.70, "shield_hit.wav": 0.75,
    "shield_break.wav": 0.85, "impact.wav": 0.95, "hurt.wav": 0.85, "warn.wav": 0.60, "eattack.wav": 0.70,
    "beam.wav": 0.95, "charge_start.wav": 0.70, "fizzle.wav": 0.50, "charge.wav": 0.55, "kicharge.wav": 0.65,
    "transform.wav": 0.95, "powerdown.wav": 0.60, "rage.wav": 0.90, "ultimate.wav": 0.85,
    "ultimate_beam.wav": 1.00, "ready.wav": 0.60, "victory.wav": 0.80, "defeat.wav": 0.70,
    "music.wav": 0.78, "music_menu.wav": 0.70,
}


def make_loop(x: np.ndarray, xfade: float = 0.05) -> np.ndarray:
    """Cross-fade the tail into the head so the loop point is click-free."""
    m = int(xfade * SR)
    if m * 2 >= len(x):
        return x
    out = x[:-m].copy()
    w = np.linspace(0.0, 1.0, m)
    out[:m] = out[:m] * w + x[-m:] * (1.0 - w)
    return out


def render(filename: str) -> np.ndarray:
    x = GENERATORS[filename]()
    if filename.startswith("music"):
        x = make_loop(np.clip(x, -1, 1), 0.08)     # already normalised by finish()
    elif filename in LOOP_FILES:
        x = make_loop(finish(x, 1.0, 1.1), 0.05)
    else:
        x = finish(x, 1.0, 1.05)
    return x * LEVEL.get(filename, 0.8)


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate_all(directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    done = []
    for name in GENERATORS:
        write_wav(directory / name, render(name))
        done.append(name)
    (directory / MARKER).write_text(PACK_VERSION)
    return done


def ensure_sound_pack(directory: Path | None = None) -> bool:
    """Generate the pack once (and again whenever PACK_VERSION changes). Never raises. Returns True if it (re)generated."""
    try:
        if directory is None:
            import config as C
            directory = C.SOUND_DIR
        marker = directory / MARKER
        ok = marker.exists() and marker.read_text().strip() == PACK_VERSION and all((directory / n).exists() for n in GENERATORS)
        if ok:
            return False
        log.info("Generating sound pack %s (first run only, a few seconds)...", PACK_VERSION)
        generate_all(directory)
        return True
    except Exception as exc:  # audio must never stop the game from starting
        log.error("Could not generate sound pack: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import config as _C
    names = generate_all(_C.SOUND_DIR)
    print(f"wrote {len(names)} files to {_C.SOUND_DIR}")