"""Layered 2D rocky battlefield: dusk sky, parallax cliffs, cracked rock floor, drifting dust,
and an additive light that follows the player's aura so energy attacks light the arena.

Everything static is pre-rendered once; per frame cost is a few blits plus ~40 dust motes.
"""
from __future__ import annotations

import math
import random

import pygame

HORIZON = 500


def _ridge(rng, w, base, amp, rough=6):
    pts = [(0, base)]
    xs = [0]
    n = 2 ** rough
    heights = [rng.uniform(-amp, amp) for _ in range(n + 1)]
    for _ in range(3):  # smooth
        heights = [heights[0]] + [(heights[i - 1] + 2 * heights[i] + heights[i + 1]) / 4 for i in range(1, n)] + [heights[-1]]
    pts = [(i * w / n, base + heights[i]) for i in range(n + 1)]
    return pts


class RockyArena:
    def __init__(self, size=(1280, 720), seed: int = 11) -> None:
        self.w, self.h = size
        rng = random.Random(seed)
        # sky gradient
        self.sky = pygame.Surface(size)
        stops = [(0.0, (22, 20, 56)), (0.45, (92, 52, 112)), (0.75, (214, 112, 100)), (1.0, (255, 176, 104))]
        for y in range(HORIZON + 20):
            f = y / (HORIZON + 20)
            for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
                if f0 <= f <= f1:
                    t = (f - f0) / (f1 - f0)
                    col = tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
                    break
            pygame.draw.line(self.sky, col, (0, y), (self.w, y))
        # sun / moon glow
        for r, c in ((150, (255, 190, 130)), (100, (255, 214, 160)), (64, (255, 240, 205))):
            pygame.draw.circle(self.sky, c, (900, HORIZON - 120), r)
        # parallax cliff layers (wider than the screen so they can slide)
        self.layers = []
        for i, (base, amp, col, speed) in enumerate(((HORIZON - 60, 70, (96, 62, 104), 4.0), (HORIZON - 20, 60, (74, 46, 84), 9.0),
                                                      (HORIZON + 10, 40, (52, 32, 62), 16.0))):
            surf = pygame.Surface((self.w + 120, self.h), pygame.SRCALPHA)
            pts = _ridge(rng, self.w + 120, base, amp)
            poly = pts + [(self.w + 120, self.h), (0, self.h)]
            pygame.draw.polygon(surf, col, poly)
            pygame.draw.lines(surf, tuple(min(255, c + 40) for c in col), False, pts, 3)
            self.layers.append((surf, speed))
        # ground
        self.ground = pygame.Surface(size, pygame.SRCALPHA)
        for y in range(HORIZON, self.h):
            f = (y - HORIZON) / (self.h - HORIZON)
            col = (int(86 - 38 * f), int(58 - 28 * f), int(56 - 26 * f), 255)
            pygame.draw.line(self.ground, col, (0, y), (self.w, y))
        for _ in range(26):  # cracks
            x, y = rng.uniform(0, self.w), rng.uniform(HORIZON + 30, self.h)
            pts = [(x, y)]
            for _ in range(4):
                x += rng.uniform(-60, 60)
                y += rng.uniform(6, 30)
                pts.append((x, y))
            pygame.draw.lines(self.ground, (24, 16, 22, 255), False, pts, 2)
        for _ in range(16):  # foreground boulders
            x, y = rng.uniform(0, self.w), rng.uniform(HORIZON + 80, self.h)
            r = rng.uniform(10, 34)
            pygame.draw.ellipse(self.ground, (34, 24, 32, 255), (x - r, y - r * 0.55, r * 2, r * 1.1))
            pygame.draw.ellipse(self.ground, (70, 50, 62, 255), (x - r * 0.8, y - r * 0.5, r * 1.5, r * 0.8))
        # dust motes (deterministic drift)
        self.motes = [(rng.uniform(0, self.w), rng.uniform(HORIZON - 120, self.h), rng.uniform(8, 30), rng.uniform(0, 6.28),
                       rng.randint(1, 3)) for _ in range(44)]
        # radial light used for energy illumination
        self._light = pygame.Surface((96, 96))
        for r in range(48, 0, -2):
            v = int(255 * (1 - r / 48) ** 1.6)
            pygame.draw.circle(self._light, (v, v, v), (48, 48), r)

    def draw(self, surf: pygame.Surface, t: float, light=None) -> None:
        """``light`` = (rgb colour, intensity 0..1.5, (x, y)) from the player's aura, or None."""
        surf.blit(self.sky, (0, 0))
        for layer, speed in self.layers:
            off = -int(60 + math.sin(t * 0.05) * speed * 2)
            surf.blit(layer, (off, 0))
        surf.blit(self.ground, (0, 0))
        for x, y, s, ph, sz in self.motes:
            px = (x + t * s) % (self.w + 40) - 20
            py = y + math.sin(t * 0.8 + ph) * 8
            c = 150 + int(60 * math.sin(t + ph))
            pygame.draw.circle(surf, (c, int(c * 0.82), int(c * 0.74)), (int(px), int(py)), sz)
        if light is not None and light[1] > 0.05:
            col, k, (lx, ly) = light
            k = min(1.5, k)
            size = int(520 + 260 * k)
            tint = self._light.copy()
            tint.fill((int(col[0] * k * 0.75), int(col[1] * k * 0.75), int(col[2] * k * 0.75)), special_flags=pygame.BLEND_RGB_MULT)
            big = pygame.transform.smoothscale(tint, (size, size))
            surf.blit(big, (int(lx - size / 2), int(ly - size / 2)), special_flags=pygame.BLEND_RGB_ADD)