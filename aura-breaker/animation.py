"""Animation helpers and procedural character artwork.

The enemy ("VOID WARDEN") and the no-camera player avatar are drawn
procedurally (original artwork). To replace the enemy with your own sprite,
drop a transparent PNG at ``assets/images/enemy.png``; it is picked up
automatically and animated with the same bob / lunge / flash / fade logic.
"""
from __future__ import annotations

import logging
import math

import pygame

import config as C
from battle_system import Battle, EnemyPhase

log = logging.getLogger("aura.anim")


# --------------------------------------------------------------------------
# Easing / tween
# --------------------------------------------------------------------------
def clamp01(t: float) -> float:
    return 0.0 if t < 0 else 1.0 if t > 1 else t


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - clamp01(t)) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 3 * t * t - 2 * t * t * t


def ease_out_back(t: float) -> float:
    t = clamp01(t)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class Tween:
    """Time-based 0..1 progress with an easing function."""

    def __init__(self, duration: float, ease=ease_out_cubic) -> None:
        self.duration = max(1e-3, duration)
        self.ease = ease
        self.t = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def update(self, dt: float) -> float:
        self.t = min(self.duration, self.t + dt)
        return self.value

    @property
    def value(self) -> float:
        return self.ease(self.t / self.duration)

    @property
    def done(self) -> bool:
        return self.t >= self.duration


# --------------------------------------------------------------------------
# Enemy
# --------------------------------------------------------------------------
def _build_enemy_body(w: int = 380, h: int = 480) -> pygame.Surface:
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    cx = w // 2

    def mirror(points):
        return [(2 * cx - x, y) for x, y in points]

    def both(color, points, outline=None, ow=3):
        for pts in (points, mirror(points)):
            pygame.draw.polygon(surf, color, pts)
            if outline:
                pygame.draw.polygon(surf, outline, pts, ow)

    dark, mid, edge = (28, 24, 48), (55, 45, 95), (150, 120, 235)
    # flowing cloak / lower body
    both((40, 18, 70), [(cx, 250), (cx - 120, 330), (cx - 150, 420), (cx - 95, 392), (cx - 70, 460), (cx - 30, 410), (cx, 470)], (120, 60, 190))
    # back spikes
    both((70, 30, 120), [(cx - 70, 150), (cx - 175, 60), (cx - 120, 160)], (140, 80, 220))
    both((70, 30, 120), [(cx - 50, 120), (cx - 120, 20), (cx - 85, 130)], (140, 80, 220))
    # torso
    both(mid, [(cx, 140), (cx - 95, 160), (cx - 80, 280), (cx - 30, 320), (cx, 310)], edge)
    # shoulder pads
    both(dark, [(cx - 70, 145), (cx - 160, 120), (cx - 190, 90), (cx - 175, 175), (cx - 105, 200)], edge)
    both((120, 90, 210), [(cx - 150, 125), (cx - 188, 92), (cx - 172, 150)])
    # arms
    both(dark, [(cx - 105, 195), (cx - 150, 280), (cx - 125, 300), (cx - 80, 215)], edge)
    both((200, 60, 60), [(cx - 150, 275), (cx - 168, 318), (cx - 128, 322), (cx - 122, 296)], (255, 150, 120))
    # head + helmet
    pygame.draw.ellipse(surf, dark, (cx - 52, 52, 104, 118))
    pygame.draw.ellipse(surf, edge, (cx - 52, 52, 104, 118), 3)
    both(dark, [(cx - 40, 70), (cx - 82, 18), (cx - 20, 52)], edge)  # horns
    both((60, 50, 100), [(cx - 10, 54), (cx - 20, 12), (cx - 2, 50)], edge)
    # visor
    pygame.draw.polygon(surf, (255, 60, 70), [(cx - 38, 104), (cx, 114), (cx + 38, 104), (cx + 30, 122), (cx, 128), (cx - 30, 122)])
    pygame.draw.polygon(surf, (255, 200, 180), [(cx - 24, 108), (cx, 115), (cx + 24, 108), (cx + 18, 115), (cx, 119), (cx - 18, 115)])
    # chest core socket
    pygame.draw.circle(surf, (20, 14, 36), (cx, 215), 30)
    pygame.draw.circle(surf, edge, (cx, 215), 30, 3)
    return surf


class EnemyRenderer:
    NAME = "VOID WARDEN"

    def __init__(self) -> None:
        self.t = 0.0
        self.defeat_t = 0.0
        self.sprite: pygame.Surface | None = None
        if C.ENEMY_SPRITE_PATH.exists():
            try:
                self.sprite = pygame.image.load(str(C.ENEMY_SPRITE_PATH)).convert_alpha()
            except pygame.error as exc:
                log.warning("Could not load enemy sprite %s: %s", C.ENEMY_SPRITE_PATH, exc)
        self.base = self.sprite if self.sprite is not None else _build_enemy_body()
        self.lunge = 0.0

    def reset(self) -> None:
        self.defeat_t = 0.0
        self.lunge = 0.0

    def update(self, dt: float, battle: Battle | None) -> None:
        self.t += dt
        if battle is None:
            return
        if battle.enemy_phase == EnemyPhase.DEFEATED:
            self.defeat_t += dt
        else:
            self.defeat_t = 0.0
        target = 1.0 if battle.enemy_phase == EnemyPhase.ATTACK else 0.0
        self.lunge += (target - self.lunge) * (1 - math.exp(-dt * 14))

    def draw(self, world: pygame.Surface, pos, battle: Battle | None) -> None:
        x, y = pos
        bob = math.sin(self.t * 2.2) * 9
        phase = battle.enemy_phase if battle else EnemyPhase.IDLE
        shake = 0.0
        glow = 0.0
        if battle is not None and phase == EnemyPhase.TELEGRAPH:
            total = max(0.1, battle.stats.telegraph)
            glow = 1.0 - max(0.0, battle.enemy_timer) / total
            shake = 2.0 + 4.0 * glow
        hit = battle.enemy_hit_timer / C.ENEMY_HIT_TIME if battle else 0.0
        dx = -60 * self.lunge + 28 * hit + math.sin(self.t * 70) * shake
        dy = bob + math.cos(self.t * 63) * shake * 0.6
        alpha = 255
        sink = 0.0
        if phase == EnemyPhase.DEFEATED:
            k = clamp01(self.defeat_t / 1.8)
            alpha = int(255 * (1 - k))
            sink = 70 * k
        # shadow
        sh = pygame.Surface((260, 50), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, int(90 * alpha / 255)), sh.get_rect())
        world.blit(sh, sh.get_rect(center=(x + dx * 0.3, y + 235)))
        body = self.base.copy()
        bw, bh = body.get_size()
        if self.sprite is not None:
            scale = 460 / bh
            body = pygame.transform.smoothscale(body, (int(bw * scale), 460))
        if hit > 0:
            v = int(190 * hit)
            body.fill((v, v, v, 0), special_flags=pygame.BLEND_RGB_ADD)
        body.set_alpha(alpha)
        rect = body.get_rect(center=(x + dx, y + dy + sink))
        world.blit(body, rect)
        core = (rect.centerx, rect.top + int(rect.height * 0.45))
        pulse = 0.6 + 0.4 * math.sin(self.t * 5)
        strength = max(glow, 0.0)
        r = int(14 + 12 * pulse + 40 * strength)
        orb = pygame.Surface((r * 4, r * 4))
        c = (r * 2, r * 2)
        base_col = (255, 60 + int(120 * (1 - strength)), 60)
        for rr, k in ((r * 2, 0.25), (int(r * 1.4), 0.5), (r, 0.9), (max(2, r // 2), 1.0)):
            pygame.draw.circle(orb, (int(base_col[0] * k), int(base_col[1] * k), int(base_col[2] * k)), c, rr)
        if alpha < 255:
            orb.fill((alpha, alpha, alpha), special_flags=pygame.BLEND_RGB_MULT)
        world.blit(orb, orb.get_rect(center=core), special_flags=pygame.BLEND_RGB_ADD)
        if strength > 0.02:  # telegraph: warning ring around the enemy
            ring = pygame.Surface((520, 520), pygame.SRCALPHA)
            a = int(200 * (0.4 + 0.6 * math.sin(self.t * 30) ** 2) * strength)
            pygame.draw.circle(ring, (255, 70, 60, a), (260, 260), int(250 - 70 * strength), 4)
            world.blit(ring, ring.get_rect(center=(rect.centerx, rect.centery)))


# --------------------------------------------------------------------------
# Fallback player avatar (used when no camera frame is available)
# --------------------------------------------------------------------------
class AvatarRenderer:
    """Simple original fighter that mirrors gesture scores, so keyboard-only
    and no-camera play still has a character to attach auras and hair to."""

    def draw(self, surf: pygame.Surface, anchor, t: float, brow_up: float = 0.0, brow_down: float = 0.0,
             mouth: float = 0.0, purse: float = 0.0, plain_hair: bool = True) -> None:
        hx, hy = anchor.head
        fw, fh = anchor.face_w, anchor.face_h
        bob = math.sin(t * 2.0) * 4
        hy += bob
        # torso
        torso = [(hx - fw * 0.55, hy + fh * 0.62), (hx + fw * 0.55, hy + fh * 0.62), (hx + fw * 1.25, hy + fh * 1.6),
                 (hx + fw * 1.4, hy + fh * 3.2), (hx - fw * 1.4, hy + fh * 3.2), (hx - fw * 1.25, hy + fh * 1.6)]
        pygame.draw.polygon(surf, (235, 110, 20), torso)
        pygame.draw.polygon(surf, (30, 60, 150), [(hx - fw * 0.3, hy + fh * 0.7), (hx + fw * 0.3, hy + fh * 0.7),
                                                   (hx + fw * 0.5, hy + fh * 3.2), (hx - fw * 0.5, hy + fh * 3.2)])
        pygame.draw.polygon(surf, (255, 170, 80), torso, 4)
        # neck + head
        pygame.draw.rect(surf, (214, 160, 125), (hx - fw * 0.2, hy + fh * 0.35, fw * 0.4, fh * 0.35))
        pygame.draw.ellipse(surf, (232, 180, 140), (hx - fw / 2, hy - fh / 2, fw, fh))
        if plain_hair:
            pygame.draw.polygon(surf, (30, 25, 35), [(hx - fw * 0.52, hy - fh * 0.05), (hx - fw * 0.45, hy - fh * 0.55),
                                                     (hx - fw * 0.2, hy - fh * 0.72), (hx, hy - fh * 0.6),
                                                     (hx + fw * 0.2, hy - fh * 0.72), (hx + fw * 0.45, hy - fh * 0.55),
                                                     (hx + fw * 0.52, hy - fh * 0.05), (hx + fw * 0.3, hy - fh * 0.3),
                                                     (hx - fw * 0.3, hy - fh * 0.3)])
        ey = hy - fh * 0.04
        for sx in (-1, 1):
            ex = hx + sx * fw * 0.22
            by = ey - fh * (0.11 + 0.05 * brow_up - 0.025 * brow_down)
            tilt = fh * 0.05 * brow_down
            pygame.draw.line(surf, (30, 25, 35), (ex - fw * 0.1, by - sx * tilt), (ex + fw * 0.1, by + sx * tilt), 5)
            pygame.draw.ellipse(surf, (255, 255, 255), (ex - fw * 0.07, ey - fh * 0.03, fw * 0.14, fh * 0.06))
            pygame.draw.circle(surf, (40, 30, 30), (int(ex), int(ey)), max(2, int(fw * 0.025)))
        my = hy + fh * 0.24
        if mouth > 0.1:
            pygame.draw.ellipse(surf, (90, 20, 30), (hx - fw * 0.13, my - fh * 0.02, fw * 0.26, fh * (0.04 + 0.12 * min(1.0, mouth))))
        elif purse > 0.3:
            pygame.draw.ellipse(surf, (170, 70, 80), (hx - fw * 0.05, my, fw * 0.1, fh * 0.05))
        else:
            pygame.draw.line(surf, (150, 70, 70), (hx - fw * 0.1, my + fh * 0.02), (hx + fw * 0.1, my + fh * 0.02), 4)
