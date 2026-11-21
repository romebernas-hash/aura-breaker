"""Original Frieza-inspired alien tyrant ("VOID EMPEROR"), drawn procedurally, facing LEFT toward the player.

Plug-in for ``animation.EnemyRenderer``: it only replaces the base body surface, so the
existing enemy phases / telegraph glow / hit flash / defeat fade all keep working.
Drop a transparent PNG at ``assets/images/enemy.png`` to override this with your own sprite.
"""
from __future__ import annotations

import math

import pygame

OUT = (38, 24, 56)
SKIN = (236, 230, 248)
SKIN_SH = (190, 180, 218)
PURPLE = (120, 64, 182)
PURPLE_DK = (72, 38, 112)
GEM = (255, 80, 175)
EYE = (232, 30, 62)


def _limb(surf, pts, width, color, ow=4):
    for col, w in ((OUT, width + ow), (color, width)):
        for p, q in zip(pts, pts[1:]):
            pygame.draw.line(surf, col, p, q, int(w))
        for p in pts:
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), int(w / 2))


def _poly(surf, color, pts, outline=OUT, ow=3):
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, outline, pts, ow)


def build_tyrant_body(w: int = 380, h: int = 480) -> pygame.Surface:
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    # tail (behind everything): curved taper to the right
    pts = []
    for i in range(21):
        t = i / 20
        x = (1 - t) ** 2 * 238 + 2 * t * (1 - t) * 370 + t * t * 356
        y = (1 - t) ** 2 * 338 + 2 * t * (1 - t) * 462 + t * t * 318
        pts.append((x, y, 30 - 22 * t))
    for col, extra in ((OUT, 4), (SKIN_SH, 0)):
        for x, y, r in pts:
            pygame.draw.circle(s, col, (int(x), int(y)), int(r / 2 + extra / 2))
    for x, y, r in pts[16:]:
        pygame.draw.circle(s, PURPLE, (int(x), int(y)), int(r / 2))
    # back arm (hanging)
    _limb(s, [(238, 178), (266, 244), (252, 304)], 28, SKIN_SH)
    _limb(s, [(262, 262), (252, 304)], 32, PURPLE_DK)
    pygame.draw.circle(s, SKIN_SH, (250, 312), 14)
    # legs
    _limb(s, [(208, 332), (236, 398), (252, 460)], 38, SKIN_SH)
    _poly(s, PURPLE_DK, [(232, 446), (290, 452), (292, 472), (228, 472)])
    _limb(s, [(182, 336), (150, 398), (126, 460)], 42, SKIN)
    _limb(s, [(158, 392), (150, 410)], 46, PURPLE)
    _poly(s, PURPLE, [(84, 448), (150, 444), (156, 472), (80, 474)])
    # torso + armour
    _poly(s, SKIN, [(148, 170), (234, 160), (244, 252), (226, 342), (170, 348), (148, 262)])
    _poly(s, PURPLE, [(148, 172), (234, 160), (240, 214), (194, 236), (150, 214)])
    _poly(s, PURPLE_DK, [(160, 300), (230, 296), (228, 330), (166, 336)])
    pygame.draw.circle(s, OUT, (192, 206), 17)
    pygame.draw.circle(s, GEM, (192, 206), 13)
    # shoulders (domes)
    for cx, cy, r in ((236, 166, 32), (144, 170, 34)):
        pygame.draw.circle(s, OUT, (cx, cy), r + 3)
        pygame.draw.circle(s, PURPLE, (cx, cy), r)
        pygame.draw.circle(s, (160, 110, 220), (cx - 8, cy - 10), r // 3)
    # front arm raised toward the player with an open palm
    _limb(s, [(146, 178), (98, 218), (46, 204)], 32, SKIN)
    _limb(s, [(84, 212), (52, 205)], 36, PURPLE)
    pygame.draw.circle(s, OUT, (34, 202), 19)
    pygame.draw.circle(s, SKIN, (34, 202), 16)
    for a in (-0.55, -0.2, 0.15, 0.5):
        ex, ey = 34 - math.cos(a) * 30, 202 + math.sin(a) * 30
        pygame.draw.line(s, OUT, (34, 202), (ex, ey), 9)
        pygame.draw.line(s, SKIN, (34, 202), (ex, ey), 5)
    pygame.draw.circle(s, GEM, (34, 202), 7)
    # neck + head
    _limb(s, [(176, 150), (178, 122)], 34, SKIN_SH)
    pygame.draw.ellipse(s, OUT, (118, 52, 94, 112))
    pygame.draw.ellipse(s, SKIN, (121, 55, 88, 106))
    # chin point + cheek
    _poly(s, SKIN, [(124, 128), (136, 156), (164, 160), (180, 140)], ow=2)
    # purple dome on the top/back of the head
    _poly(s, PURPLE, [(210, 112), (206, 74), (176, 50), (140, 54), (126, 76), (152, 74), (182, 86), (196, 118)])
    # horns
    _poly(s, SKIN, [(158, 60), (146, 10), (176, 54)], ow=3)
    _poly(s, SKIN, [(184, 66), (206, 18), (200, 76)], ow=3)
    # face (looking left)
    pygame.draw.polygon(s, EYE, [(128, 96), (158, 90), (152, 108), (130, 108)])
    pygame.draw.polygon(s, OUT, [(126, 94), (160, 86), (160, 92), (128, 100)])
    pygame.draw.line(s, OUT, (126, 134), (156, 138), 4)
    pygame.draw.line(s, OUT, (122, 118), (130, 124), 3)
    return s