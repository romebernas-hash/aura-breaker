"""Real-time visual effects engine.

Layers are drawn onto a black "glow" surface that is bloomed and then added
(BLEND_RGB_ADD) onto the world, so every energy effect has a real glow.

Everything is driven by delta time and clipped to safe coordinates. Particle
counts are capped by the quality setting.

Changes in this version:
  * beams, the charge orb and the ultimate orb come from ``anchor.hand`` (the fighter's firing hand)
    when it is set, otherwise from the mouth / above the head as before;
  * KI CHARGE (power-up) aura, inward-gathering particles, ground dust and a light shake;
  * ``ki_charge_start`` and ``ultimate_ready`` events.

Note on the aura: no body segmentation is used. The aura is attached to an
ellipse estimated from the player anchor (head + body approximation).
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

import config as C
from battle_system import Battle, Event
from face_tracking import PlayerAnchor
from game_state import GameState as S

log = logging.getLogger("aura.fx")

Color = tuple[int, int, int]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def P(x: float, y: float) -> tuple[int, int]:
    """Sanitise a coordinate so pygame never receives NaN/inf/huge values."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return (0, 0)
    return (int(clamp(x, -4000, 6000)), int(clamp(y, -4000, 6000)))


def scale_color(c: Color, k: float) -> Color:
    k = clamp(k, 0.0, 1.5)
    return (int(clamp(c[0] * k, 0, 255)), int(clamp(c[1] * k, 0, 255)), int(clamp(c[2] * k, 0, 255)))


def ease_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


@dataclass(frozen=True)
class AuraStyle:
    name: str
    outer: Color
    mid: Color
    core: Color


AURA_STYLES: dict[str, AuraStyle] = {
    "gold": AuraStyle("gold", (255, 150, 10), (255, 205, 50), (255, 245, 180)),
    "blue": AuraStyle("blue", (20, 90, 255), (70, 170, 255), (200, 240, 255)),
    "red": AuraStyle("red", (190, 10, 20), (255, 60, 40), (255, 190, 150)),
    "purple": AuraStyle("purple", (110, 30, 220), (180, 90, 255), (235, 200, 255)),
    "white_gold": AuraStyle("white_gold", (255, 200, 90), (255, 235, 170), (255, 255, 245)),
}


@dataclass
class FxContext:
    anchor: PlayerAnchor
    enemy_pos: tuple[float, float]
    battle: Battle | None = None


def orb_point(anchor: PlayerAnchor) -> tuple[float, float]:
    """Where the ultimate energy sphere forms: the cupped hands, or above the head without a fighter."""
    hand = getattr(anchor, "hand", None)
    if hand is not None:
        return hand
    return (anchor.head[0], anchor.head[1] - anchor.face_h * 1.5)


def aim_geometry(anchor: PlayerAnchor, enemy_pos: tuple[float, float], aim: float):
    """Return (muzzle, angle_rad, distance_to_enemy_from_muzzle).

    ``aim`` in [-1, 1], positive = up. At aim 0 the beam points at the enemy.
    The beam starts at the fighter's hand when ``anchor.hand`` is set, else at the mouth.
    """
    hand = getattr(anchor, "hand", None)
    if hand is not None:
        ox, oy = hand
        reach = anchor.face_w * 0.25
    else:
        ox, oy = anchor.mouth[0] + anchor.face_w * 0.15, anchor.mouth[1]
        reach = anchor.face_w * 0.8
    base = math.atan2(enemy_pos[1] - oy, enemy_pos[0] - ox)
    angle = base - math.radians(C.MAX_AIM_DEG) * clamp(aim, -1, 1)
    muzzle = (ox + math.cos(angle) * reach, oy + math.sin(angle) * reach)
    dist = math.dist(muzzle, enemy_pos)
    return muzzle, angle, dist


# --------------------------------------------------------------------------
# Particles (numpy-backed pool)
# --------------------------------------------------------------------------
class ParticleSystem:
    def __init__(self, capacity: int) -> None:
        self.set_capacity(capacity)

    def set_capacity(self, capacity: int) -> None:
        n = max(8, int(capacity))
        self.capacity = n
        self.pos = np.zeros((n, 2), np.float32)
        self.vel = np.zeros((n, 2), np.float32)
        self.tgt = np.zeros((n, 2), np.float32)
        self.life = np.zeros(n, np.float32)
        self.max_life = np.ones(n, np.float32)
        self.size = np.ones(n, np.float32)
        self.color = np.zeros((n, 3), np.float32)
        self.attract = np.zeros(n, np.float32)
        self.drag = np.zeros(n, np.float32)
        self.gravity = np.zeros(n, np.float32)
        self.alive = np.zeros(n, bool)

    @property
    def count(self) -> int:
        return int(self.alive.sum())

    def clear(self) -> None:
        self.alive[:] = False

    def emit(
        self,
        n: int,
        pos,
        speed=(40.0, 140.0),
        angle=(0.0, math.tau),
        life=(0.5, 1.0),
        size=(2.0, 5.0),
        color: Color = (255, 200, 60),
        color2: Color | None = None,
        attract: float = 0.0,
        target=None,
        drag: float = 0.0,
        gravity: float = 0.0,
        pos_jitter: float = 0.0,
        rng: random.Random | np.random.Generator | None = None,
    ) -> int:
        """Spawn up to ``n`` particles. Returns how many were actually spawned
        (the pool is a hard cap, extra requests are dropped)."""
        if n <= 0:
            return 0
        free = np.flatnonzero(~self.alive)[:n]
        k = len(free)
        if k == 0:
            return 0
        g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        ang = g.uniform(angle[0], angle[1], k)
        spd = g.uniform(speed[0], speed[1], k)
        p0 = np.asarray(pos, np.float32)
        self.pos[free] = p0 + g.uniform(-pos_jitter, pos_jitter, (k, 2)) if pos_jitter else p0
        self.vel[free, 0] = np.cos(ang) * spd
        self.vel[free, 1] = np.sin(ang) * spd
        lf = g.uniform(life[0], life[1], k)
        self.life[free] = lf
        self.max_life[free] = lf
        self.size[free] = g.uniform(size[0], size[1], k)
        if color2 is None:
            self.color[free] = color
        else:
            mix = g.uniform(0, 1, (k, 1))
            self.color[free] = np.array(color) * (1 - mix) + np.array(color2) * mix
        self.attract[free] = attract
        self.tgt[free] = np.asarray(target if target is not None else p0, np.float32)
        self.drag[free] = drag
        self.gravity[free] = gravity
        self.alive[free] = True
        return k

    def update(self, dt: float) -> None:
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return
        self.life[idx] -= dt
        dead = idx[self.life[idx] <= 0]
        self.alive[dead] = False
        idx = idx[self.life[idx] > 0]
        if idx.size == 0:
            return
        att = self.attract[idx]
        if np.any(att != 0):
            self.vel[idx] += (self.tgt[idx] - self.pos[idx]) * att[:, None] * dt
        self.vel[idx, 1] += self.gravity[idx] * dt
        self.vel[idx] *= np.clip(1.0 - self.drag[idx] * dt, 0.0, 1.0)[:, None]
        self.pos[idx] += self.vel[idx] * dt
        np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def render(self, surf: pygame.Surface) -> None:
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return
        w, h = surf.get_size()
        a = (self.life[idx] / self.max_life[idx]).tolist()
        xs = self.pos[idx, 0].tolist()
        ys = self.pos[idx, 1].tolist()
        sz = self.size[idx].tolist()
        cols = self.color[idx].tolist()
        for i in range(idx.size):
            x, y = xs[i], ys[i]
            if x < -30 or x > w + 30 or y < -30 or y > h + 30:
                continue
            al = a[i]
            c = cols[i]
            col = (int(c[0] * al), int(c[1] * al), int(c[2] * al))
            pygame.draw.circle(surf, col, (int(x), int(y)), max(1, int(sz[i] * (0.4 + 0.6 * al))))


# --------------------------------------------------------------------------
# Lightning
# --------------------------------------------------------------------------
def lightning_path(start, end, rng: random.Random, jag: float = 0.22, depth: int = 4) -> list[tuple[float, float]]:
    """Midpoint-displacement lightning. Endpoints are preserved exactly."""
    pts = [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
    length = math.dist(pts[0], pts[1])
    offset = length * jag
    for _ in range(max(1, depth)):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            seg = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / seg, dx / seg
            o = rng.uniform(-offset, offset)
            out.append(((a[0] + b[0]) / 2 + nx * o, (a[1] + b[1]) / 2 + ny * o))
            out.append(b)
        pts = out
        offset *= 0.5
    return pts


class LightningField:
    """A handful of bolts around the aura ellipse, regenerated every few frames."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.bolts: list[tuple[list[tuple[float, float]], float, float]] = []  # path, brightness, width scale
        self.timer = 0.0

    def clear(self) -> None:
        self.bolts.clear()

    def update(self, dt: float, center, radii, intensity: float, max_bolts: int) -> None:
        self.timer -= dt
        if self.timer > 0:
            return
        self.timer = self.rng.uniform(0.04, 0.09)
        self.bolts = []
        count = int(round(max_bolts * clamp(intensity, 0.0, 1.2)))
        cx, cy = center
        rx, ry = radii
        for _ in range(count):
            ang = self.rng.uniform(math.radians(-215), math.radians(35))  # upper arc + sides
            sx, sy = cx + math.cos(ang) * rx * 0.95, cy + math.sin(ang) * ry * 0.95
            nx, ny = math.cos(ang), math.sin(ang)
            length = self.rng.uniform(0.35, 0.8) * min(rx, ry) * 0.9
            tang = self.rng.uniform(-0.5, 0.5)
            ex, ey = sx + (nx - ny * tang) * length, sy + (ny + nx * tang) * length
            path = lightning_path((sx, sy), (ex, ey), self.rng)
            br = self.rng.uniform(0.6, 1.0)
            self.bolts.append((path, br, 1.0))
            if self.rng.random() < 0.5:  # branch
                k = self.rng.randrange(2, max(3, len(path) - 2))
                bx, by = path[k]
                ba = ang + self.rng.uniform(-1.0, 1.0)
                bl = length * self.rng.uniform(0.3, 0.55)
                self.bolts.append((lightning_path((bx, by), (bx + math.cos(ba) * bl, by + math.sin(ba) * bl), self.rng, depth=3), br * 0.7, 0.6))

    def render(self, surf: pygame.Surface, style: AuraStyle) -> None:
        for path, br, ws in self.bolts:
            pts = [P(x, y) for x, y in path]
            if len(pts) < 2:
                continue
            pygame.draw.lines(surf, scale_color(style.outer, 0.55 * br), False, pts, max(2, int(8 * ws)))
            pygame.draw.lines(surf, scale_color(style.mid, br), False, pts, max(1, int(4 * ws)))
            pygame.draw.lines(surf, scale_color((255, 255, 255), br), False, pts, max(1, int(2 * ws)))


# --------------------------------------------------------------------------
# Aura
# --------------------------------------------------------------------------
def draw_aura(surf: pygame.Surface, style: AuraStyle, center, radii, intensity: float, t: float, tongues: int) -> None:
    """Layered ring glow + animated flame tongues around an ellipse.

    The ring is hollow so it frames the player instead of washing out the face.
    """
    inten = clamp(intensity, 0.0, 1.5)
    if inten < 0.02:
        return
    cx, cy = center
    rx, ry = radii
    pulse = 1.0 + 0.04 * math.sin(t * 5.1) + 0.02 * math.sin(t * 11.3)
    # glow rings (outermost first, dimmer -> brighter inward)
    layers = 5
    for k in range(layers):
        f = k / (layers - 1)
        grow = 1.0 + (1 - f) * 0.20
        w2, h2 = rx * grow * pulse * 2, ry * grow * pulse * 2
        rect = pygame.Rect(0, 0, int(clamp(w2, 4, 4000)), int(clamp(h2, 4, 4000)))
        rect.center = P(cx, cy)
        col = scale_color(style.outer if f < 0.5 else style.mid, (0.20 + 0.55 * f) * inten)
        pygame.draw.ellipse(surf, col, rect, max(2, int((18 - 12 * f) * (0.6 + 0.4 * inten))))
    # flame tongues
    base_h = 0.34 * ry * (0.6 + 0.5 * inten)
    for i in range(tongues):
        f = i / max(1, tongues - 1)
        ang = math.radians(-205 + 235 * f)
        ca, sa = math.cos(ang), math.sin(ang)
        bx, by = cx + ca * rx * pulse, cy + sa * ry * pulse
        nx, ny = ca / max(rx, 1), sa / max(ry, 1)
        nl = math.hypot(nx, ny) or 1
        nx, ny = nx / nl, ny / nl
        dx, dy = nx, ny - 0.85  # flames lean upward
        dl = math.hypot(dx, dy) or 1
        dx, dy = dx / dl, dy / dl
        tx, ty = -ny, nx
        phase = i * 1.713
        flick = 0.55 + 0.45 * math.sin(t * (6.0 + (i % 5)) + phase)
        noise = 0.75 + 0.25 * math.sin(t * 3.3 + phase * 2.1) * math.cos(t * 1.7 + phase)
        h = base_h * flick * noise * (0.7 + 0.3 * math.sin(i * 0.9))
        wbase = max(8.0, (2 * math.pi * min(rx, ry)) / max(8, tongues) * 0.42)
        sway = math.sin(t * 4.2 + phase) * 0.35 * h
        for scale, col in ((1.0, style.outer), (0.72, style.mid), (0.42, style.core)):
            hh, ww = h * scale, wbase * (0.5 + 0.5 * scale)
            tip = (bx + dx * hh + tx * sway * scale, by + dy * hh + ty * sway * scale)
            pts = [P(bx - tx * ww, by - ty * ww), P(*tip), P(bx + tx * ww, by + ty * ww)]
            pygame.draw.polygon(surf, scale_color(col, 0.35 + 0.65 * min(1.0, inten)), pts)


# --------------------------------------------------------------------------
# Beams, projectiles, shockwaves
# --------------------------------------------------------------------------
class Beam:
    def __init__(self, origin, angle: float, speed: float, impact_dist: float | None, style: AuraStyle, width: float, life: float) -> None:
        self.origin = origin
        self.angle = angle
        self.dir = (math.cos(angle), math.sin(angle))
        self.speed = max(100.0, speed)
        self.max_len = impact_dist if impact_dist is not None else 2600.0
        self.style = style
        self.width = width
        self.life = life
        self.t = 0.0
        self.head = 0.0
        self.impact_fired = impact_dist is None

    @property
    def alive(self) -> bool:
        return self.t < self.life

    def update(self, dt: float) -> bool:
        """Advance. Returns True on the frame the beam head reaches its impact point."""
        self.t += dt
        self.head = min(self.max_len, self.head + self.speed * dt)
        if not self.impact_fired and self.head >= self.max_len:
            self.impact_fired = True
            return True
        return False

    def point_at(self, dist: float):
        return (self.origin[0] + self.dir[0] * dist, self.origin[1] + self.dir[1] * dist)

    def render(self, surf: pygame.Surface) -> None:
        fade = 1.0 - clamp((self.t - (self.life - 0.2)) / 0.2, 0.0, 1.0)
        if fade <= 0 or self.head <= 1:
            return
        pulse = 1.0 + 0.12 * math.sin(self.t * 60)
        p0, p1 = self.origin, self.point_at(self.head)
        nx, ny = -self.dir[1], self.dir[0]
        grow = ease_out(self.t / 0.12)
        for wf, col, br in ((1.0, self.style.outer, 0.55), (0.62, self.style.mid, 0.9), (0.26, (255, 255, 255), 1.0)):
            w0 = self.width * wf * 0.45 * grow * pulse
            w1 = self.width * wf * 0.9 * grow * pulse
            pts = [P(p0[0] + nx * w0, p0[1] + ny * w0), P(p1[0] + nx * w1, p1[1] + ny * w1),
                   P(p1[0] - nx * w1, p1[1] - ny * w1), P(p0[0] - nx * w0, p0[1] - ny * w0)]
            pygame.draw.polygon(surf, scale_color(col, br * fade), pts)
        # traveling energy beads
        spacing = 70.0
        off = (self.t * 1400.0) % spacing
        d = off
        while d < self.head:
            bx, by = self.point_at(d)
            pygame.draw.circle(surf, scale_color((255, 255, 255), fade), P(bx, by), max(2, int(self.width * 0.12)))
            d += spacing
        pygame.draw.circle(surf, scale_color(self.style.mid, fade), P(*p1), max(4, int(self.width * 0.7 * grow)))
        pygame.draw.circle(surf, scale_color((255, 255, 255), fade), P(*p1), max(3, int(self.width * 0.35 * grow)))
        if self.t < 0.18:  # muzzle flash
            r = int(self.width * (1.3 - self.t * 4))
            pygame.draw.circle(surf, scale_color(self.style.mid, 1 - self.t / 0.18), P(*p0), max(2, r))
            pygame.draw.circle(surf, scale_color((255, 255, 255), 1 - self.t / 0.18), P(*p0), max(2, r // 2))


class Projectile:
    """Enemy energy ball travelling toward the player."""

    def __init__(self, start, end, duration: float, style: AuraStyle) -> None:
        self.start, self.end, self.duration, self.style = start, end, max(0.05, duration), style
        self.t = 0.0

    @property
    def alive(self) -> bool:
        return self.t < self.duration

    @property
    def pos(self):
        k = clamp(self.t / self.duration, 0, 1)
        return (self.start[0] + (self.end[0] - self.start[0]) * k, self.start[1] + (self.end[1] - self.start[1]) * k)

    def update(self, dt: float) -> bool:
        before = self.t
        self.t += dt
        return before < self.duration <= self.t

    def render(self, surf: pygame.Surface) -> None:
        x, y = self.pos
        pulse = 1 + 0.15 * math.sin(self.t * 40)
        pygame.draw.circle(surf, scale_color(self.style.outer, 0.7), P(x, y), int(38 * pulse))
        pygame.draw.circle(surf, scale_color(self.style.mid, 0.95), P(x, y), int(26 * pulse))
        pygame.draw.circle(surf, scale_color(self.style.core, 1.0), P(x, y), int(13 * pulse))


class Shockwave:
    def __init__(self, pos, color: Color, max_r: float, life: float = 0.6, width: int = 10) -> None:
        self.pos, self.color, self.max_r, self.life, self.width, self.t = pos, color, max_r, life, width, 0.0

    @property
    def alive(self) -> bool:
        return self.t < self.life

    def update(self, dt: float) -> None:
        self.t += dt

    def render(self, surf: pygame.Surface) -> None:
        k = clamp(self.t / self.life, 0, 1)
        r = max(2, int(self.max_r * ease_out(k)))
        w = max(1, int(self.width * (1 - k)))
        pygame.draw.circle(surf, scale_color(self.color, 1 - k), P(*self.pos), r, w)


class ScreenFx:
    """Flash, dim and shake with low-effects scaling."""

    def __init__(self) -> None:
        self.flash = 0.0
        self.dim = 0.0
        self.dim_target = 0.0
        self.shake = 0.0
        self.scale = 1.0   # lowered in low-effects mode

    def trigger_flash(self, v: float) -> None:
        self.flash = max(self.flash, clamp(v * self.scale, 0, 1))

    def add_shake(self, mag: float) -> None:
        self.shake = max(self.shake, mag * self.scale)

    def set_dim(self, target: float) -> None:
        self.dim_target = clamp(target, 0, 0.85)

    def update(self, dt: float) -> None:
        self.flash *= math.exp(-dt * 6.0)
        if self.flash < 0.01:
            self.flash = 0.0
        self.dim += (self.dim_target - self.dim) * (1 - math.exp(-dt * 5.0))
        self.shake *= math.exp(-dt * 7.0)
        if self.shake < 0.2:
            self.shake = 0.0

    def offset(self, rng: random.Random) -> tuple[int, int]:
        if self.shake <= 0:
            return (0, 0)
        return (int(rng.uniform(-1, 1) * self.shake), int(rng.uniform(-1, 1) * self.shake))


# --------------------------------------------------------------------------
# Hair overlay
# --------------------------------------------------------------------------
def make_hair_surface(w: int = 800, h: int = 560, ss: int = 2) -> pygame.Surface:
    """Original procedural spiky-hair artwork (also used to generate the PNG).

    The hairline anchor is at (0.5 * w, 0.74 * h).
    """
    W, H = w * ss, h * ss
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    cx, cy = W * 0.5, H * 0.74
    rx, ry = W * 0.36, H * 0.22
    spikes = [  # (angle deg, x-scale, y-scale, tilt deg)
        (166, 1.16, 1.75, 8), (146, 1.10, 2.25, 6), (126, 1.04, 2.65, 4), (106, 1.0, 2.95, 1),
        (84, 1.0, 3.05, -1), (62, 1.04, 2.7, -4), (42, 1.10, 2.3, -6), (22, 1.16, 1.8, -8),
    ]
    pts: list[tuple[float, float]] = [(cx - rx * 0.90, cy + H * 0.15), (cx - rx * 1.03, cy + H * 0.0)]
    prev_ang = 190.0
    for ang, sx, sy, tilt in spikes:
        mid = math.radians((prev_ang + ang) / 2)
        pts.append((cx + math.cos(mid) * rx * 0.97, cy - math.sin(mid) * ry * 1.3))
        a = math.radians(ang + tilt)
        pts.append((cx + math.cos(a) * rx * sx, cy - math.sin(a) * ry * sy))
        prev_ang = ang
    mid = math.radians((prev_ang - 5) / 2)
    pts.append((cx + math.cos(mid) * rx * 0.97, cy - math.sin(mid) * ry * 1.3))
    pts.append((cx + rx * 1.03, cy + H * 0.0))
    pts.append((cx + rx * 0.90, cy + H * 0.15))
    # bangs (right -> left)
    bangs = [(0.62, 0.09, 0.52), (0.42, 0.05, 0.32), (0.22, 0.12, 0.12), (0.02, 0.06, -0.06),
             (-0.18, 0.12, -0.26), (-0.38, 0.05, -0.46), (-0.58, 0.10, -0.66)]
    for fx, fy, valley_x in bangs:
        pts.append((cx + rx * valley_x, cy + H * 0.02))
        pts.append((cx + rx * fx, cy + H * (0.02 + fy)))
    pts.append((cx - rx * 0.78, cy + H * 0.02))

    def scaled(k: float):
        return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in pts]

    layers = [(1.0, (205, 120, 10)), (0.90, (255, 190, 30)), (0.74, (255, 222, 80)), (0.55, (255, 244, 150))]
    for k, col in layers:
        pygame.draw.polygon(surf, col, scaled(k))
    pygame.draw.polygon(surf, (150, 80, 0), pts, max(2, 5 * ss))
    for ang, sx, sy, tilt in spikes[1:-1:2]:  # highlight streaks
        a = math.radians(ang + tilt)
        pygame.draw.line(surf, (255, 252, 210), (cx + math.cos(a) * rx * 0.55, cy - math.sin(a) * ry * 1.0),
                         (cx + math.cos(a) * rx * sx * 0.85, cy - math.sin(a) * ry * sy * 0.85), max(2, 4 * ss))
    return pygame.transform.smoothscale(surf, (w, h))


class HairOverlay:
    """Transparent-PNG hair anchored to forehead landmarks (scale/rotation tracked).

    Falls back to a procedurally drawn version of the same artwork if the PNG is
    missing (logged once). Large head turns or occlusion reduce alignment accuracy.
    Only used in the full-webcam view; the side-view fighter draws its own hair.
    """

    ANCHOR = (0.5, 0.74)

    def __init__(self, path: Path = C.HAIR_PATH) -> None:
        self.using_fallback = False
        try:
            img = pygame.image.load(str(path))
            img = img.convert_alpha() if pygame.display.get_surface() else img
        except (pygame.error, FileNotFoundError, OSError) as exc:
            log.warning("Hair asset %s unavailable (%s). Using procedural fallback.", path, exc)
            img = make_hair_surface()
            self.using_fallback = True
        scale = 520 / img.get_width()
        self.image = pygame.transform.smoothscale(img, (520, max(1, int(img.get_height() * scale))))

    def draw(self, surf: pygame.Surface, anchor: PlayerAnchor, alpha: float, tint: str | None = None) -> None:
        if alpha <= 0.01:
            return
        iw, ih = self.image.get_size()
        scale = clamp(anchor.face_w * 1.48 / iw, 0.05, 6.0)
        local = ((self.ANCHOR[0] - 0.5) * iw * scale, (self.ANCHOR[1] - 0.5) * ih * scale)
        ang = anchor.roll_deg
        rot = pygame.transform.rotozoom(self.image, ang, scale)
        a = math.radians(ang)
        ox = local[0] * math.cos(a) + local[1] * math.sin(a)
        oy = -local[0] * math.sin(a) + local[1] * math.cos(a)
        if tint == "red":
            rot.fill((255, 150, 130, 255), special_flags=pygame.BLEND_RGBA_MULT)
        elif tint == "white":
            rot.fill((70, 70, 70, 0), special_flags=pygame.BLEND_RGB_ADD)
        rot.set_alpha(int(255 * clamp(alpha, 0, 1)))
        center = (anchor.hairline[0] - ox, anchor.hairline[1] - oy)
        surf.blit(rot, rot.get_rect(center=P(*center)))


# --------------------------------------------------------------------------
# Effects manager
# --------------------------------------------------------------------------
class Fx:
    def __init__(self, quality: str = "standard", size: tuple[int, int] = (C.LOGICAL_W, C.LOGICAL_H), seed: int | None = None) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.glow = pygame.Surface(size).convert() if pygame.display.get_surface() else pygame.Surface(size)
        self.particles = ParticleSystem(100)
        self.lightning = LightningField(self.rng)
        self.screen = ScreenFx()
        self.beams: list[Beam] = []
        self.projectiles: list[Projectile] = []
        self.shockwaves: list[Shockwave] = []
        self.ripples: list[list[float]] = []   # [angle, age]
        self.t = 0.0
        self.aura_style = AURA_STYLES["blue"]
        self.aura_intensity = 0.0
        self.hair_alpha = 0.0
        self.hair_target = 0.0
        self.shield_alpha = 0.0
        self._acc_aura = 0.0
        self._acc_charge = 0.0
        self._acc_beam = 0.0
        self._acc_dust = 0.0
        self.set_quality(quality)

    # -- configuration -----------------------------------------------------
    def set_quality(self, quality: str) -> None:
        self.quality = quality if quality in C.QUALITY else "standard"
        self.q = C.QUALITY[self.quality]
        self.particles.set_capacity(self.q["particles"])
        self.screen.scale = 0.5 if self.quality == "low" else 1.0
        if self.quality == "low":
            self.lightning.clear()

    def reset(self) -> None:
        self.particles.clear()
        self.beams.clear()
        self.projectiles.clear()
        self.shockwaves.clear()
        self.ripples.clear()
        self.lightning.clear()
        self.screen.flash = 0.0
        self.screen.dim = self.screen.dim_target = 0.0
        self.screen.shake = 0.0
        self.aura_intensity = 0.0
        self.hair_alpha = self.hair_target = 0.0
        self.shield_alpha = 0.0

    # -- events ------------------------------------------------------------
    def burst(self, pos, color: Color, color2: Color, n: int, speed=(120, 420), life=(0.4, 1.0), size=(2, 6)) -> None:
        n = int(n * self.q["spawn"])
        self.particles.emit(n, pos, speed=speed, life=life, size=size, color=color, color2=color2, drag=1.5, rng=self.np_rng)

    def handle_event(self, ev: Event, ctx: FxContext) -> None:
        k, d = ev.kind, ev.data
        a = ctx.anchor
        sc = self.screen
        gold, red, wg, blue = AURA_STYLES["gold"], AURA_STYLES["red"], AURA_STYLES["white_gold"], AURA_STYLES["blue"]
        if k == "transform_start":
            sc.set_dim(0.55)
            sc.add_shake(4)
        elif k == "transform_complete":
            sc.set_dim(0.0)
            sc.trigger_flash(1.0)
            sc.add_shake(10)
            self.hair_target = 1.0
            c = a.aura_center
            for i, r in enumerate((260, 420, 600)):
                self.shockwaves.append(Shockwave(c, gold.mid, r, 0.7 + 0.15 * i, 14 - 3 * i))
            self.burst(c, gold.mid, gold.core, 140, speed=(150, 520))
        elif k == "transform_end":
            self.hair_target = 0.0
            sc.set_dim(0.0)
        elif k == "rage_start":
            sc.add_shake(7)
            sc.set_dim(0.3)
        elif k == "rage_active":
            sc.set_dim(0.0)
            sc.trigger_flash(0.5)
            self.shockwaves.append(Shockwave(a.aura_center, red.mid, 420, 0.6, 12))
            self.burst(a.aura_center, red.outer, red.core, 90, speed=(150, 450))
        elif k == "ki_charge_start":
            self.shockwaves.append(Shockwave(a.aura_center, blue.mid, 200, 0.5, 8))
            sc.add_shake(2)
        elif k == "ultimate_ready":
            sc.trigger_flash(0.25)
            self.shockwaves.append(Shockwave(a.aura_center, wg.mid, 280, 0.7, 8))
            self.burst(a.aura_center, wg.mid, wg.core, 40, speed=(80, 260))
        elif k == "beam_fire":
            self._spawn_beam(ctx, d.get("aim", 0.0), d.get("hit", True), d.get("charge", 0.5), ultimate=False)
            sc.add_shake(2 + 5 * d.get("charge", 0.5))
        elif k == "ultimate_start":
            sc.set_dim(0.6)
        elif k == "ultimate_fire":
            self._spawn_beam(ctx, 0.0, True, 1.0, ultimate=True)
            sc.trigger_flash(0.8)
            sc.add_shake(12)
        elif k == "ultimate_end":
            sc.set_dim(0.0)
        elif k == "enemy_hit":
            ult = d.get("source") == "ultimate"
            dmg = d.get("damage", 10)
            style = wg if ult else (red if ctx.battle is not None and ctx.battle.rage_time > 0 else (gold if ctx.battle is not None and ctx.battle.transformed else blue))
            self.shockwaves.append(Shockwave(ctx.enemy_pos, style.mid, 300 if ult else 150 + dmg * 3, 0.55, 12))
            if ult:
                self.shockwaves.append(Shockwave(ctx.enemy_pos, (255, 255, 255), 520, 0.9, 16))
                sc.trigger_flash(1.0)
            self.burst(ctx.enemy_pos, style.mid, style.core, int(30 + dmg * (3 if ult else 1.6)))
            sc.add_shake(min(14, 3 + dmg * 0.18))
        elif k == "enemy_attack":
            self.projectiles.append(Projectile(ctx.enemy_pos, a.head, d.get("travel", C.ENEMY_PROJECTILE_TIME), red))
        elif k == "player_hit":
            if d.get("damage", 0) > 0:
                self.shockwaves.append(Shockwave(a.head, red.mid, 190, 0.45, 10))
                self.burst(a.head, red.mid, red.core, 40)
                sc.add_shake(7)
                sc.trigger_flash(0.25)
        elif k == "shield_hit":
            self.ripples.append([self.rng.uniform(-0.8, 0.8), 0.0])  # enemy side (right)
            sc.add_shake(3)
        elif k == "shield_break":
            self.shockwaves.append(Shockwave(a.head, blue.core, 300, 0.5, 8))
            self.burst(a.head, blue.mid, blue.core, 60)
        elif k == "victory":
            sc.trigger_flash(0.8)
            self.shockwaves.append(Shockwave(ctx.enemy_pos, (255, 255, 255), 700, 1.0, 16))
            self.burst(ctx.enemy_pos, gold.mid, gold.core, 200)
        elif k == "defeat":
            sc.set_dim(0.45)

    def _spawn_beam(self, ctx: FxContext, aim: float, hit: bool, charge: float, ultimate: bool) -> None:
        muzzle, angle, dist = aim_geometry(ctx.anchor, ctx.enemy_pos, aim)
        if hit:  # visually snap to the enemy; the hit test uses the aim tolerance
            angle = math.atan2(ctx.enemy_pos[1] - muzzle[1], ctx.enemy_pos[0] - muzzle[0])
        impact = max(40.0, dist - C.ENEMY_RADIUS * 0.7) if hit else None
        travel = C.BEAM_TRAVEL_TIME if not ultimate else 0.30
        ref = impact if impact is not None else dist
        speed = ref / travel
        b = ctx.battle
        if ultimate:
            style = AURA_STYLES["white_gold"]
        elif b is not None and b.rage_time > 0:
            style = AURA_STYLES["red"]
        elif b is not None and b.transformed:
            style = AURA_STYLES["gold"]
        else:
            style = AURA_STYLES["blue"]
        width = 150.0 if ultimate else 34.0 + 48.0 * clamp(charge, 0, 1)
        self.beams.append(Beam(muzzle, angle, speed, impact, style, width, 1.1 if ultimate else C.BEAM_DURATION))

    # -- state-driven aura target -----------------------------------------
    def _aura_target(self, b: Battle | None) -> tuple[str, float]:
        if b is None:
            return "blue", 0.0
        st = b.fsm.state
        ef = clamp(b.energy / C.PLAYER_MAX_ENERGY, 0, 1)
        ki = bool(getattr(b, "ki_charging", False))
        kib = (0.45 + 0.12 * math.sin(self.t * 14)) if ki else 0.0   # power-up pulse
        if st == S.ULTIMATE:
            return "white_gold", 0.5 + 1.0 * clamp(b.ult_time / C.ULTIMATE_CHARGE_TIME, 0, 1)
        if st == S.RAGE_BURST:
            return "red", 0.4 + 0.9 * (1 - clamp(b.anim_time / C.RAGE_ANIM, 0, 1))
        if b.rage_time > 0:
            return "red", 0.75 + 0.35 * math.sin(self.t * 6) * 0.3 + 0.2 * ef + kib
        if st == S.TRANSFORMING:
            return "gold", 0.3 + 1.1 * (1 - clamp(b.anim_time / C.TRANSFORM_ANIM, 0, 1))
        if b.transformed:
            return "gold", 0.30 + 0.70 * ef + (0.4 * b.charge if st == S.CHARGING else 0.0) + kib
        if st == S.CHARGING:
            return "purple", 0.3 + 0.9 * b.charge
        if ki:
            return "blue", 0.55 + kib
        return "blue", 0.16 if self.quality != "low" else 0.0

    # -- update ------------------------------------------------------------
    def update(self, dt: float, ctx: FxContext) -> None:
        dt = clamp(dt, 0.0, 0.1)
        self.t += dt
        b = ctx.battle
        a = ctx.anchor
        q = self.q
        sc = self.screen
        name, target = self._aura_target(b)
        self.aura_style = AURA_STYLES[name]
        self.aura_intensity += (target - self.aura_intensity) * (1 - math.exp(-dt / 0.12))
        self.hair_alpha += (self.hair_target - self.hair_alpha) * (1 - math.exp(-dt / 0.25))
        want_shield = 1.0 if (b is not None and b.shield_active) else 0.0
        self.shield_alpha += (want_shield - self.shield_alpha) * (1 - math.exp(-dt / 0.10))
        sc.update(dt)

        center, radii = a.aura_center, a.aura_radii
        inten = self.aura_intensity
        spawn = q["spawn"]

        # aura particles rising from the aura edge
        if inten > 0.25:
            self._acc_aura += 70 * inten * spawn * dt
            n = int(self._acc_aura)
            self._acc_aura -= n
            if n:
                ang = self.np_rng.uniform(math.radians(-200), math.radians(20), n)
                for i in range(n):
                    px = center[0] + math.cos(ang[i]) * radii[0] * 0.95
                    py = center[1] + math.sin(ang[i]) * radii[1] * 0.95
                    self.particles.emit(1, (px, py), speed=(20, 70), angle=(math.radians(-115), math.radians(-65)),
                                        life=(0.6, 1.3), size=(2, 5), color=self.aura_style.mid, color2=self.aura_style.core,
                                        drag=0.3, gravity=-30, rng=self.np_rng)

        # lightning (only for energetic styles, kept at the aura edge)
        if q["bolts"] and name in ("gold", "red", "white_gold", "purple") and inten > 0.45:
            self.lightning.update(dt, center, radii, min(1.0, (inten - 0.3) / 0.9), q["bolts"])
        else:
            self.lightning.clear()

        # charging particles gather at the muzzle
        if b is not None:
            muzzle, _, _ = aim_geometry(a, ctx.enemy_pos, b.aim)
            if b.fsm.state == S.CHARGING:
                self._acc_charge += 160 * (0.3 + b.charge) * spawn * dt
                n = int(self._acc_charge)
                self._acc_charge -= n
                for _ in range(n):
                    ang = self.rng.uniform(0, math.tau)
                    r = self.rng.uniform(90, 190)
                    pos = (muzzle[0] + math.cos(ang) * r, muzzle[1] + math.sin(ang) * r)
                    self.particles.emit(1, pos, speed=(0, 10), life=(0.3, 0.5), size=(2, 4), color=self.aura_style.mid,
                                        color2=(255, 255, 255), attract=9.0, target=muzzle, drag=1.2, rng=self.np_rng)
            elif b.fsm.state == S.ULTIMATE and b.ult_time < C.ULTIMATE_CHARGE_TIME:
                self._acc_charge += 260 * spawn * dt
                n = int(self._acc_charge)
                self._acc_charge -= n
                orb = orb_point(a)
                for _ in range(n):
                    ang = self.rng.uniform(0, math.tau)
                    r = self.rng.uniform(200, 380)
                    pos = (orb[0] + math.cos(ang) * r, orb[1] + math.sin(ang) * r)
                    self.particles.emit(1, pos, speed=(0, 10), life=(0.4, 0.7), size=(2, 5), color=(255, 235, 170),
                                        color2=(255, 255, 255), attract=7.0, target=orb, drag=1.0, rng=self.np_rng)
            elif b.fsm.state == S.TRANSFORMING:
                self._acc_charge += 140 * spawn * dt
                n = int(self._acc_charge)
                self._acc_charge -= n
                for _ in range(n):
                    pos = (center[0] + self.rng.uniform(-1, 1) * radii[0] * 1.3, center[1] + radii[1] * 1.1)
                    self.particles.emit(1, pos, speed=(0, 20), angle=(math.radians(-100), math.radians(-80)),
                                        life=(0.8, 1.4), size=(2, 6), color=(255, 190, 30), color2=(255, 250, 190),
                                        gravity=-160, rng=self.np_rng)
            elif getattr(b, "ki_charging", False):
                # KI CHARGE: energy streams inward to the body, dust kicks up at the feet, the ground trembles
                self._acc_charge += 230 * spawn * dt
                n = int(self._acc_charge)
                self._acc_charge -= n
                for _ in range(n):
                    ang = self.rng.uniform(0, math.tau)
                    r = self.rng.uniform(150, 320)
                    pos = (center[0] + math.cos(ang) * r * 0.8, center[1] + math.sin(ang) * r * 1.1)
                    self.particles.emit(1, pos, speed=(0, 10), life=(0.35, 0.6), size=(2, 5), color=self.aura_style.mid,
                                        color2=(255, 255, 255), attract=7.0, target=center, drag=0.9, rng=self.np_rng)
                self._acc_dust += 60 * spawn * dt
                nd = int(self._acc_dust)
                self._acc_dust -= nd
                foot = (a.head[0], a.head[1] + a.face_h * 2.16)
                for _ in range(nd):
                    self.particles.emit(1, foot, speed=(40, 170), angle=(math.radians(-170), math.radians(-10)),
                                        life=(0.4, 0.9), size=(2, 4), color=(190, 160, 130), color2=(120, 100, 90),
                                        gravity=-10, drag=1.6, pos_jitter=40.0, rng=self.np_rng)
                sc.add_shake(1.5)

        # beam trail particles + impact detection
        for beam in self.beams:
            reached = beam.update(dt)
            if beam.alive and beam.head > 10:
                self._acc_beam += 140 * spawn * dt
                n = int(self._acc_beam)
                self._acc_beam -= n
                for _ in range(n):
                    p = beam.point_at(self.rng.uniform(0, beam.head))
                    self.particles.emit(1, p, speed=(20, 120), life=(0.25, 0.5), size=(2, 4), color=beam.style.mid,
                                        color2=(255, 255, 255), drag=2.0, rng=self.np_rng)
            if reached:
                end = beam.point_at(beam.head)
                self.burst(end, beam.style.mid, beam.style.core, 24, speed=(80, 300))
        self.beams = [bm for bm in self.beams if bm.alive]

        for pr in self.projectiles:
            arrived = pr.update(dt)
            if arrived:
                self.burst(pr.end, pr.style.mid, pr.style.core, 16, speed=(60, 220))
            elif self.rng.random() < 0.8:
                x, y = pr.pos
                self.particles.emit(1, (x, y), speed=(10, 60), life=(0.2, 0.45), size=(2, 5), color=pr.style.mid,
                                    color2=pr.style.core, drag=2.0, rng=self.np_rng)
        self.projectiles = [p for p in self.projectiles if p.alive]

        for sw in self.shockwaves:
            sw.update(dt)
        self.shockwaves = [s for s in self.shockwaves if s.alive]
        for r in self.ripples:
            r[1] += dt
        self.ripples = [r for r in self.ripples if r[1] < 0.6]
        self.particles.update(dt)

    # -- drawing -----------------------------------------------------------
    def draw_dim(self, world: pygame.Surface) -> None:
        if self.screen.dim > 0.01:
            m = int(255 * (1 - self.screen.dim))
            world.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)

    def shake_offset(self) -> tuple[int, int]:
        return self.screen.offset(self.rng)

    def _bloom(self, passes: int) -> None:
        if passes <= 0:
            return
        w, h = self.size
        small = pygame.transform.smoothscale(self.glow, (max(1, w // 4), max(1, h // 4)))
        self.glow.blit(pygame.transform.smoothscale(small, (w, h)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if passes >= 2:
            tiny = pygame.transform.smoothscale(small, (max(1, w // 10), max(1, h // 10)))
            self.glow.blit(pygame.transform.smoothscale(tiny, (w, h)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def _shield_geometry(self, a: PlayerAnchor):
        return (a.head[0], a.head[1] + a.face_h * 0.55), a.face_h * 1.75

    def draw_shield_fill(self, world: pygame.Surface, ctx: FxContext) -> None:
        if self.shield_alpha < 0.02 or ctx.battle is None:
            return
        (cx, cy), r = self._shield_geometry(ctx.anchor)
        r = int(clamp(r, 20, 1500))
        frac = clamp(ctx.battle.shield_strength / C.SHIELD_MAX_STRENGTH, 0, 1)
        col = (60, 160, 255) if frac > 0.35 else (255, 140, 60)
        size = r * 2 + 4
        tmp = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*col, int(46 * self.shield_alpha)), (r + 2, r + 2), r)
        pygame.draw.circle(tmp, (*col, int(110 * self.shield_alpha)), (r + 2, r + 2), r, 4)
        world.blit(tmp, tmp.get_rect(center=P(cx, cy)))

    def draw(self, world: pygame.Surface, ctx: FxContext) -> None:
        b = ctx.battle
        a = ctx.anchor
        self.draw_shield_fill(world, ctx)
        g = self.glow
        g.fill((0, 0, 0))
        drew = False
        if self.aura_intensity > 0.03:
            draw_aura(g, self.aura_style, a.aura_center, a.aura_radii, self.aura_intensity, self.t, self.q["tongues"])
            drew = True
        if self.lightning.bolts:
            self.lightning.render(g, self.aura_style)
            drew = True
        if self.shield_alpha > 0.02 and b is not None:
            (cx, cy), r = self._shield_geometry(a)
            frac = clamp(b.shield_strength / C.SHIELD_MAX_STRENGTH, 0, 1)
            col = (90, 190, 255) if frac > 0.35 else (255, 150, 70)
            r = int(clamp(r, 20, 1500))
            pygame.draw.circle(g, scale_color(col, 0.8 * self.shield_alpha), P(cx, cy), r, 5)
            pygame.draw.circle(g, scale_color(col, 0.35 * self.shield_alpha), P(cx, cy), int(r * (0.96 + 0.02 * math.sin(self.t * 8))), 2)
            for i in range(3):  # traveling energy arcs on the rim
                ang = self.t * (1.4 + i * 0.6) + i * 2.1
                rect = pygame.Rect(0, 0, r * 2, r * 2)
                rect.center = P(cx, cy)
                pygame.draw.arc(g, scale_color((220, 245, 255), self.shield_alpha), rect, ang, ang + 0.7, 4)
            for ang0, age in self.ripples:
                k = age / 0.6
                rr = int(r * (0.85 + 0.25 * k))
                rect = pygame.Rect(0, 0, rr * 2, rr * 2)
                rect.center = P(cx, cy)
                pygame.draw.arc(g, scale_color((255, 255, 255), 1 - k), rect, ang0 - 0.6, ang0 + 0.6, 7)
            drew = True
        # charge sphere / ultimate orb
        if b is not None:
            st = b.fsm.state
            if st == S.CHARGING and b.charge > 0.01:
                muzzle, _, _ = aim_geometry(a, ctx.enemy_pos, b.aim)
                self._draw_orb(g, muzzle, 12 + 62 * b.charge, self.aura_style)
                drew = True
            elif st == S.ULTIMATE and b.ult_time < C.ULTIMATE_CHARGE_TIME + 0.2:
                orb = orb_point(a)
                k = clamp(b.ult_time / C.ULTIMATE_CHARGE_TIME, 0, 1)
                self._draw_orb(g, orb, 20 + 150 * k * (1.0 if b.ult_time < C.ULTIMATE_CHARGE_TIME else 0.6), AURA_STYLES["white_gold"])
                drew = True
        for beam in self.beams:
            beam.render(g)
            drew = True
        for pr in self.projectiles:
            pr.render(g)
            drew = True
        for sw in self.shockwaves:
            sw.render(g)
            drew = True
        if self.particles.count:
            self.particles.render(g)
            drew = True
        if drew:
            self._bloom(self.q["bloom"])
            world.blit(g, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if self.screen.flash > 0.01:
            v = int(255 * self.screen.flash)
            world.fill((v, v, v), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_orb(self, surf: pygame.Surface, pos, radius: float, style: AuraStyle) -> None:
        pulse = 1 + 0.08 * math.sin(self.t * 18)
        r = radius * pulse
        pygame.draw.circle(surf, scale_color(style.outer, 0.6), P(*pos), int(r * 1.5))
        pygame.draw.circle(surf, scale_color(style.mid, 0.9), P(*pos), int(r))
        pygame.draw.circle(surf, scale_color(style.core, 1.0), P(*pos), int(r * 0.6))
        pygame.draw.circle(surf, (255, 255, 255), P(*pos), int(r * 0.3))
        for i in range(4):  # orbiting sparks
            ang = self.t * 5 + i * math.tau / 4
            pygame.draw.circle(surf, (255, 255, 255), P(pos[0] + math.cos(ang) * r * 1.8, pos[1] + math.sin(ang) * r * 1.8), 3)