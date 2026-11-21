"""Pygame user interface: menus, HUD, scanner, calibration, overlays."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pygame

import config as C
from animation import ease_out_back, ease_out_cubic
from battle_system import ABILITY_LABEL, Ability, Battle
from face_controls import CALIB_PROMPTS, GESTURE_LABEL, CalibStep, Calibrator, Gesture, GestureFrame
from game_state import GameState as S

W, H = C.LOGICAL_W, C.LOGICAL_H
WHITE = (240, 244, 255)
GOLD = (255, 205, 60)
BLUE = (80, 175, 255)
RED = (255, 70, 60)
PURPLE = (180, 100, 255)
GREEN = (80, 230, 140)
GREY = (130, 140, 165)
PANEL = (10, 14, 28)

_font_cache: dict[int, pygame.font.Font] = {}
_text_cache: dict[tuple, pygame.Surface] = {}


def font(size: int) -> pygame.font.Font:
    f = _font_cache.get(size)
    if f is None:
        f = pygame.font.Font(None, size)  # bundled freesansbold
        _font_cache[size] = f
    return f


def _render(s: str, size: int, color) -> pygame.Surface:
    key = (s, size, tuple(color))
    img = _text_cache.get(key)
    if img is None:
        if len(_text_cache) > 600:
            _text_cache.clear()
        img = font(size).render(s, True, color)
        _text_cache[key] = img
    return img


def text(surf, s: str, pos, size: int = 26, color=WHITE, align: str = "left", shadow: bool = True, alpha: int = 255):
    img = _render(str(s), size, color)
    r = img.get_rect()
    anchor = {"left": "topleft", "center": "center", "right": "topright", "midtop": "midtop", "midleft": "midleft", "midright": "midright"}[align]
    setattr(r, anchor, pos)
    if alpha < 255:
        img = img.copy()
        img.set_alpha(alpha)
    if shadow:
        sh = _render(str(s), size, (0, 0, 0))
        if alpha < 255:
            sh = sh.copy()
            sh.set_alpha(alpha)
        surf.blit(sh, r.move(2, 2))
    surf.blit(img, r)
    return r


def panel(surf, rect, alpha: int = 170, border=(70, 90, 140), radius: int = 8) -> None:
    rect = pygame.Rect(rect)
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*PANEL, alpha), s.get_rect(), border_radius=radius)
    pygame.draw.rect(s, (*border, 230), s.get_rect(), 2, border_radius=radius)
    surf.blit(s, rect.topleft)


def bar(surf, rect, frac: float, c1, c2, label: str | None = None, value: str | None = None, flash: float = 0.0) -> None:
    rect = pygame.Rect(rect)
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(surf, (8, 10, 20), rect, border_radius=5)
    fill = pygame.Rect(rect.x + 2, rect.y + 2, int((rect.w - 4) * frac), rect.h - 4)
    if fill.w > 0:
        pygame.draw.rect(surf, c1, fill, border_radius=4)
        hi = pygame.Rect(fill.x, fill.y, fill.w, max(2, fill.h // 2))
        pygame.draw.rect(surf, c2, hi, border_radius=4)
    if flash > 0:
        pygame.draw.rect(surf, (255, 255, 255), rect, 2, border_radius=5)
    pygame.draw.rect(surf, (110, 125, 170), rect, 2, border_radius=5)
    if label:
        text(surf, label, (rect.x + 8, rect.centery), 20, WHITE, "midleft")
    if value:
        text(surf, value, (rect.right - 8, rect.y + 2), 20, WHITE, "right")


# --------------------------------------------------------------------------
# Background arena (used when there is no camera frame)
# --------------------------------------------------------------------------
class ArenaBackground:
    def __init__(self) -> None:
        self.grad = pygame.Surface((W, H))
        for y in range(H):
            k = y / H
            c = (int(8 + 22 * k), int(10 + 16 * k), int(28 + 40 * (1 - abs(k - 0.55) * 1.6)))
            pygame.draw.line(self.grad, c, (0, y), (W, y))
        import random
        r = random.Random(7)
        self.stars = [(r.randrange(W), r.randrange(int(H * 0.55)), r.uniform(0.3, 1.0)) for _ in range(90)]
        self.vignette = pygame.Surface((W, H), pygame.SRCALPHA)
        for i in range(12):
            a = int(10 + 8 * i)
            pygame.draw.rect(self.vignette, (0, 0, 0, max(0, 70 - a)), (i * 6, i * 6, W - i * 12, H - i * 12), 6)

    def draw(self, surf, t: float) -> None:
        surf.blit(self.grad, (0, 0))
        for x, y, b in self.stars:
            v = int(120 + 120 * b * (0.6 + 0.4 * math.sin(t * 2 + x)))
            surf.fill((v, v, v), (x, y, 2, 2))
        horizon = int(H * 0.62)
        pygame.draw.line(surf, (60, 90, 170), (0, horizon), (W, horizon), 2)
        for i in range(1, 12):  # floor grid (perspective)
            y = horizon + int((i / 11) ** 2 * (H - horizon))
            pygame.draw.line(surf, (30, 50, 110), (0, y), (W, y), 1)
        for i in range(-14, 15):
            x0 = W / 2 + i * 40
            x1 = W / 2 + i * 190
            pygame.draw.line(surf, (30, 50, 110), (x0, horizon), (x1, H), 1)
        surf.blit(self.vignette, (0, 0))


# --------------------------------------------------------------------------
# Menus
# --------------------------------------------------------------------------
class Menu:
    def __init__(self, items: list[str], cx: int = W // 2, top: int = 330, spacing: int = 62, width: int = 440) -> None:
        self.items, self.cx, self.top, self.spacing, self.width = items, cx, top, spacing, width
        self.selected = 0
        self.age = 0.0

    def reset(self) -> None:
        self.age = 0.0
        self.selected = 0

    def rect(self, i: int) -> pygame.Rect:
        r = pygame.Rect(0, 0, self.width, 50)
        r.center = (self.cx, self.top + i * self.spacing)
        return r

    def key(self, key: int) -> int | None:
        if key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.items)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.items)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            return self.selected
        return None

    def mouse(self, pos, click: bool) -> int | None:
        for i in range(len(self.items)):
            if self.rect(i).collidepoint(pos):
                self.selected = i
                if click:
                    return i
        return None

    def draw(self, surf, dt: float, t: float) -> None:
        self.age += dt
        for i, label in enumerate(self.items):
            k = ease_out_back((self.age - i * 0.07) / 0.45)
            if k <= 0:
                continue
            r = self.rect(i)
            r.x += int((1 - k) * 180)
            sel = i == self.selected
            s = pygame.Surface(r.size, pygame.SRCALPHA)
            base = (255, 190, 40) if sel else (60, 80, 140)
            pygame.draw.rect(s, (*PANEL, 200), s.get_rect(), border_radius=10)
            pygame.draw.rect(s, (*base, 255 if sel else 150), s.get_rect(), 3 if sel else 2, border_radius=10)
            if sel:
                pygame.draw.rect(s, (255, 190, 40, int(40 + 25 * math.sin(t * 6))), s.get_rect(), border_radius=10)
            s.set_alpha(int(255 * min(1.0, k)))
            surf.blit(s, r)
            text(surf, label, r.center, 32, GOLD if sel else WHITE, "center", alpha=int(255 * min(1.0, k)))


@dataclass
class SettingRow:
    label: str
    get: Callable[[], str]
    change: Callable[[int], None]
    activate: Callable[[], None] | None = None


class SettingsScreen:
    ROW_H = 44

    def __init__(self, rows: list[SettingRow]) -> None:
        self.rows = rows
        self.selected = 0
        self.top = 0

    def _visible(self) -> int:
        return (H - 190) // self.ROW_H

    def key(self, key: int) -> str | None:
        n = len(self.rows)
        if key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % n
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % n
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.rows[self.selected].change(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.rows[self.selected].change(+1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            row = self.rows[self.selected]
            if row.activate:
                row.activate()
            else:
                row.change(+1)
        vis = self._visible()
        if self.selected < self.top:
            self.top = self.selected
        elif self.selected >= self.top + vis:
            self.top = self.selected - vis + 1
        return None

    def mouse(self, pos, click: bool, wheel: int = 0) -> None:
        if wheel:
            self.top = max(0, min(max(0, len(self.rows) - self._visible()), self.top - wheel))
        for i in range(self.top, min(len(self.rows), self.top + self._visible())):
            r = pygame.Rect(240, 120 + (i - self.top) * self.ROW_H, W - 480, self.ROW_H - 4)
            if r.collidepoint(pos):
                self.selected = i
                if click:
                    row = self.rows[i]
                    if row.activate:
                        row.activate()
                    else:
                        row.change(+1 if pos[0] > r.centerx else -1)

    def draw(self, surf, t: float) -> None:
        text(surf, "SETTINGS", (W // 2, 56), 60, GOLD, "center")
        vis = self._visible()
        for i in range(self.top, min(len(self.rows), self.top + vis)):
            row = self.rows[i]
            r = pygame.Rect(240, 120 + (i - self.top) * self.ROW_H, W - 480, self.ROW_H - 4)
            sel = i == self.selected
            panel(surf, r, 200 if sel else 130, GOLD if sel else (60, 80, 140), 6)
            text(surf, row.label, (r.x + 16, r.centery), 26, WHITE, "midleft")
            text(surf, f"<  {row.get()}  >" if row.activate is None else row.get(), (r.right - 16, r.centery), 26,
                 GOLD if sel else BLUE, "midright")
        if len(self.rows) > vis:
            text(surf, f"{self.selected + 1}/{len(self.rows)}", (W - 240, H - 92), 20, GREY, "right")
        text(surf, "UP/DOWN select   LEFT/RIGHT change   ENTER activate   ESC back", (W // 2, H - 50), 22, GREY, "center")


# --------------------------------------------------------------------------
# Toasts and banners
# --------------------------------------------------------------------------
class Banner:
    def __init__(self) -> None:
        self.title = ""
        self.sub = ""
        self.color = GOLD
        self.t = 99.0
        self.duration = 2.5

    def show(self, title: str, sub: str = "", color=GOLD, duration: float = 2.5) -> None:
        self.title, self.sub, self.color, self.duration, self.t = title, sub, color, duration, 0.0

    def update(self, dt: float) -> None:
        self.t += dt

    def draw(self, surf) -> None:
        if self.t >= self.duration:
            return
        k = ease_out_back(self.t / 0.35)
        fade = 1.0 - max(0.0, (self.t - (self.duration - 0.5)) / 0.5)
        a = int(255 * max(0, min(1, fade)))
        size = max(10, int(92 * min(1.15, k)))
        text(surf, self.title, (W // 2, 170), size, self.color, "center", alpha=a)
        if self.sub:
            text(surf, self.sub, (W // 2, 238), 38, WHITE, "center", alpha=a)


class Toasts:
    def __init__(self) -> None:
        self.items: list[list] = []

    def add(self, msg: str, color=WHITE, duration: float = 1.6) -> None:
        if self.items and self.items[-1][0] == msg and self.items[-1][1] > duration - 0.4:
            return
        self.items.append([msg, duration, color])
        self.items = self.items[-4:]

    def update(self, dt: float) -> None:
        for it in self.items:
            it[1] -= dt
        self.items = [it for it in self.items if it[1] > 0]

    def draw(self, surf) -> None:
        for i, (msg, left, color) in enumerate(reversed(self.items)):
            a = int(255 * min(1.0, left / 0.4))
            text(surf, msg, (W // 2, H - 188 - i * 32), 28, color, "center", alpha=a)


DENY_TEXT = {
    "energy": "NOT ENOUGH ENERGY",
    "cooldown": "ON COOLDOWN",
    "busy": "FINISH CURRENT ACTION",
    "active": "ALREADY ACTIVE",
    "shield_weak": "SHIELD RECHARGING",
    "not_in_battle": "NOT IN BATTLE",
}


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------
def tracking_label(pipe, frame: GestureFrame | None, face_on: bool) -> tuple[str, tuple[int, int, int]]:
    if pipe is None:
        return "KEYBOARD ONLY", GREY
    if pipe.status == "starting":
        return "STARTING CAMERA...", GOLD
    if pipe.status == "no_camera":
        return "NO CAMERA - KEYBOARD MODE", RED
    if pipe.status == "camera_lost":
        return "CAMERA LOST - KEYBOARD MODE", RED
    if pipe.tracker_status == "unavailable":
        return "NO FACE MODEL - RUN setup_assets", RED
    if not face_on:
        return "FACE CONTROLS OFF", GREY
    if frame is None or not frame.face_present:
        return "FACE LOST", RED
    if not frame.tracking_ready:
        return "HOLD NEUTRAL FACE...", GOLD
    return "FACE TRACKING OK", GREEN


def draw_gesture_panel(surf, frame: GestureFrame | None, x: int, y: int) -> None:
    panel(surf, (x, y, 250, 142), 150)
    text(surf, "GESTURES", (x + 12, y + 8), 20, GREY)
    for i, g in enumerate((Gesture.MOUTH_OPEN, Gesture.BROW_RAISE, Gesture.BROW_LOWER, Gesture.PURSE)):
        yy = y + 34 + i * 26
        score = frame.scores.get(g, 0.0) if frame else 0.0
        active = bool(frame and g in frame.active)
        col = GOLD if active else BLUE
        text(surf, GESTURE_LABEL[g], (x + 12, yy), 18, GOLD if active else WHITE)
        bar(surf, (x + 118, yy, 120, 16), min(1.0, score), col, tuple(min(255, c + 60) for c in col))


def draw_ability_bar(surf, b: Battle, t: float) -> None:
    n = 6
    w, h, gap = 130, 74, 10
    x0 = (W - (n * w + (n - 1) * gap)) // 2
    y = H - 92
    for i, a in enumerate(Ability):
        r = pygame.Rect(x0 + i * (w + gap), y, w, h)
        ready, why = b.can_use(a)
        active = (
            (a == Ability.TRANSFORM and b.transformed) or (a == Ability.RAGE and b.rage_time > 0)
            or (a == Ability.SHIELD and b.shield_active) or (a == Ability.SCANNER and b.scan_time > 0)
            or (a == Ability.BLASTER and b.fsm.state == S.CHARGING)
            or (a == Ability.ULTIMATE and b.fsm.state == S.ULTIMATE)
        )
        border = GOLD if active else (GREEN if ready else (70, 80, 120))
        if a == Ability.ULTIMATE and ready:
            border = tuple(int(c * (0.7 + 0.3 * math.sin(t * 8))) for c in (255, 235, 150))
        panel(surf, r, 190, border, 8)
        frac = b.cooldown_fraction(a)
        if frac > 0 and not active:
            ov = pygame.Surface((r.w - 4, int((r.h - 4) * frac)), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            surf.blit(ov, (r.x + 2, r.y + 2))
            text(surf, f"{b.cooldowns[a]:.0f}s", r.center, 26, GREY, "center")
        text(surf, str(int(a)), (r.x + 8, r.y + 4), 24, GOLD)
        text(surf, ABILITY_LABEL[a], (r.centerx, r.bottom - 18), 21, WHITE if ready or active else GREY, "center")
        if a == Ability.ULTIMATE and not ready and why == "energy":
            text(surf, f"{int(b.energy)}/{int(C.ULTIMATE_COST)}", (r.centerx, r.y + 26), 20, GREY, "center")
        elif active:
            text(surf, "ACTIVE", (r.centerx, r.y + 26), 20, GOLD, "center")


def draw_hud(surf, b: Battle, t: float, frame: GestureFrame | None, track: tuple[str, tuple], training: bool,
             fps: float, show_fps: bool, enemy_name: str) -> None:
    # player
    panel(surf, (16, 14, 420, 112), 160)
    text(surf, "PLAYER", (28, 20), 22, GREY)
    hp = b.player_hp / C.PLAYER_MAX_HP
    bar(surf, (28, 44, 396, 26), hp, (60, 200, 90) if hp > 0.35 else (220, 60, 50), (130, 255, 150) if hp > 0.35 else (255, 130, 110),
        "HP", f"{int(b.player_hp)}")
    ef = b.energy / C.PLAYER_MAX_ENERGY
    c1 = (230, 170, 20) if b.transformed else (40, 120, 240)
    c2 = (255, 235, 120) if b.transformed else (120, 200, 255)
    bar(surf, (28, 76, 396, 24), ef, c1, c2, "ENERGY", f"{int(b.energy)}")
    text(surf, f"POWER LEVEL {b.power:,}", (28, 104), 20, GOLD if b.transformed else WHITE)
    if b.transformed:
        text(surf, f"AWAKENED  {b.transform_time:0.0f}s", (424, 104), 20, GOLD, "right")
    elif b.rage_time > 0:
        text(surf, f"RAGE  {b.rage_time:0.0f}s", (424, 104), 20, RED, "right")
    # shield strength
    if b.shield_active or b.shield_strength < C.SHIELD_MAX_STRENGTH - 0.5:
        bar(surf, (28, 130, 200, 16), b.shield_strength / C.SHIELD_MAX_STRENGTH, (40, 140, 220), (140, 210, 255))
        text(surf, "SHIELD", (236, 128), 18, BLUE)
    # enemy
    panel(surf, (W - 436, 14, 420, 84), 160)
    text(surf, f"{enemy_name}   ROUND {b.round}" if not training else f"TRAINING DUMMY", (W - 424, 20), 22, GREY)
    eh = b.enemy_hp / b.enemy_max_hp if b.enemy_max_hp else 0
    bar(surf, (W - 424, 46, 396, 30), eh, (200, 50, 70), (255, 120, 130), None, f"{int(b.enemy_hp)}/{int(b.enemy_max_hp)}",
        flash=1.0 if b.enemy_hit_timer > 0 else 0.0)
    # score
    if not training:
        text(surf, f"SCORE {b.score:,}", (W // 2, 18), 34, WHITE, "midtop")
    if b.combo >= 2:
        mult = 1.0 + C.COMBO_STEP * b.combo
        text(surf, f"{b.combo} HIT COMBO  x{min(C.COMBO_MAX_MULT, mult):.1f}", (W // 2, 56), 30, GOLD, "midtop")
    # aim indicator
    aim_x = W // 2
    text(surf, "AIM", (aim_x - 120, H - 150), 18, GREY, "center")
    pygame.draw.line(surf, (80, 90, 130), (aim_x - 100, H - 140), (aim_x + 100, H - 140), 3)
    pygame.draw.circle(surf, GOLD, (int(aim_x + b.aim * 100), H - 140), 8)
    pygame.draw.rect(surf, (255, 255, 255), (aim_x - int(C.AIM_TOLERANCE * 100), H - 146, int(C.AIM_TOLERANCE * 200), 12), 1)
    # charge meter
    if b.fsm.state == S.CHARGING:
        bar(surf, (W // 2 - 180, H - 180, 360, 26), b.charge, (150, 70, 255), (220, 170, 255), "CHARGE", f"{int(b.charge * 100)}%")
    draw_ability_bar(surf, b, t)
    # tracking + gestures
    txt, col = track
    panel(surf, (W - 316, H - 214, 300, 36), 150, col, 8)
    text(surf, txt, (W - 166, H - 196), 22, col, "center")
    draw_gesture_panel(surf, frame, 16, H - 252)
    if show_fps:
        text(surf, f"{fps:0.0f} FPS", (W - 16, H - 24), 20, GREY, "right")


# --------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------
class ScannerReadout:
    """Eases displayed numbers toward real game values (no invented data)."""

    def __init__(self) -> None:
        self.player = 0.0
        self.enemy = 0.0

    def update(self, dt: float, b: Battle) -> None:
        k = 1 - math.exp(-dt * 6)
        self.player += (b.power - self.player) * k
        self.enemy += (b.enemy_power - self.enemy) * k


def _brackets(surf, rect, color, t, size=26, width=4) -> None:
    r = pygame.Rect(rect).inflate(int(6 * math.sin(t * 5)), int(6 * math.sin(t * 5)))
    for (px, py, dx, dy) in ((r.left, r.top, 1, 1), (r.right, r.top, -1, 1), (r.left, r.bottom, 1, -1), (r.right, r.bottom, -1, -1)):
        pygame.draw.line(surf, color, (px, py), (px + dx * size, py), width)
        pygame.draw.line(surf, color, (px, py), (px, py + dy * size), width)


def draw_scanner(surf, b: Battle, anchor, enemy_pos, t: float, ro: ScannerReadout) -> None:
    if b.scan_time <= 0:
        return
    life = C.SCAN_DURATION - b.scan_time
    fade = min(1.0, life / 0.25, b.scan_time / 0.4)
    col = (90, 255, 220)
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((0, 40, 40, int(70 * fade)))
    for y in range(0, H, 6):
        pygame.draw.line(ov, (80, 255, 220, int(28 * fade)), (0, y), (W, y))
    sy = int((t * 380) % H)
    pygame.draw.rect(ov, (120, 255, 230, int(90 * fade)), (0, sy, W, 5))
    surf.blit(ov, (0, 0))
    ex, ey = enemy_pos
    _brackets(surf, (ex - 130, ey - 190, 260, 400), RED, t)
    fw, fh = anchor.face_w, anchor.face_h
    _brackets(surf, (anchor.head[0] - fw * 0.8, anchor.head[1] - fh * 0.8, fw * 1.6, fh * 1.6), col, t)
    # enemy readout
    panel(surf, (int(ex - 300), int(ey - 150), 190, 120), 190, RED)
    text(surf, "ENEMY READING", (int(ex - 290), int(ey - 142)), 18, RED)
    text(surf, f"{int(ro.enemy):,}", (int(ex - 290), int(ey - 118)), 36, WHITE)
    text(surf, f"HP {int(b.enemy_hp)}/{int(b.enemy_max_hp)}", (int(ex - 290), int(ey - 76)), 20, WHITE)
    text(surf, "SIMULATED GAME DATA", (int(ex - 290), int(ey - 54)), 15, GREY)
    # player readout
    px, py = anchor.head[0] + fw * 0.95, anchor.head[1] - fh * 0.6
    px = min(px, W - 250)
    panel(surf, (int(px), int(py), 230, 138), 190, col)
    text(surf, "POWER SCAN", (int(px) + 10, int(py) + 8), 18, col)
    text(surf, f"{int(ro.player):,}", (int(px) + 10, int(py) + 30), 38, GOLD if b.transformed else WHITE)
    text(surf, f"ENERGY {int(b.energy)}/{int(C.PLAYER_MAX_ENERGY)}", (int(px) + 10, int(py) + 74), 20, WHITE)
    form = "RAGE" if b.rage_time > 0 else ("AWAKENED x%.1f" % C.TRANSFORM_MULT if b.transformed else "BASE FORM")
    text(surf, f"FORM {form}", (int(px) + 10, int(py) + 98), 20, WHITE)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def draw_calibration(surf, cal: Calibrator, frame: GestureFrame | None, status: tuple[str, tuple], t: float,
                     camera_ok: bool) -> None:
    panel(surf, (W // 2 - 420, 30, 840, 190), 200)
    text(surf, "FACE CALIBRATION", (W // 2, 54), 40, GOLD, "center")
    if not camera_ok:
        text(surf, "No camera available.", (W // 2, 110), 34, RED, "center")
        text(surf, "Default thresholds are used. Keyboard controls work normally.", (W // 2, 150), 26, WHITE, "center")
        text(surf, "ESC: back", (W // 2, 190), 24, GREY, "center")
        return
    step = cal.step
    title, hint = CALIB_PROMPTS[step]
    idx = list(CalibStep).index(step) + 1
    text(surf, f"STEP {idx}/{len(CalibStep)}   {title}", (W // 2, 108), 36, WHITE, "center")
    text(surf, hint, (W // 2, 148), 26, GREY, "center")
    if step != CalibStep.TEST:
        label = "GET READY..." if cal.phase == "prep" else "HOLD IT!"
        bar(surf, (W // 2 - 250, 176, 500, 26), cal.progress, GOLD if cal.phase == "record" else BLUE, (255, 240, 170), label)
        if cal.no_face_time > 1.0:
            text(surf, "NO FACE DETECTED - move into view", (W // 2, 232), 28, RED, "center")
    else:
        for i, w in enumerate(cal.warnings[:3]):
            text(surf, w, (W // 2, 232 + i * 26), 20, GOLD, "center")
        if frame is not None:
            draw_gesture_panel(surf, frame, W // 2 - 125, 330)
        text(surf, "ENTER: save    R: repeat    ESC: cancel", (W // 2, H - 60), 28, WHITE, "center")
    text(surf, status[0], (W // 2, H - 100), 24, status[1], "center")


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def dim_overlay(surf, a: int = 150) -> None:
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((0, 0, 0, a))
    surf.blit(ov, (0, 0))


def draw_result(surf, victory: bool, b: Battle, t: float, menu: Menu, dt: float) -> None:
    dim_overlay(surf, 160)
    col = GOLD if victory else RED
    pulse = 1 + 0.03 * math.sin(t * 4)
    text(surf, "VICTORY!" if victory else "DEFEATED", (W // 2, 150), int(120 * pulse), col, "center")
    text(surf, f"SCORE {b.score:,}   ROUND {b.round}", (W // 2, 235), 40, WHITE, "center")
    if victory:
        text(surf, f"Next enemy is stronger. Power +{int(C.UPGRADE_POWER_BONUS * 100)}%", (W // 2, 278), 26, GREY, "center")
    menu.top = 360
    menu.draw(surf, dt, t)


def draw_controls_help(surf, t: float) -> None:
    dim_overlay(surf, 200)
    text(surf, "CONTROLS", (W // 2, 50), 56, GOLD, "center")
    rows = [
        ("FACE", "Brows down  -> Transform / Rage (AUTO priority)"),
        ("FACE", "Open mouth (hold) -> charge; close to fire"),
        ("FACE", "Brows up (hold) -> Power Scanner"),
        ("FACE", "Brows up + mouth open (hold) -> Ultimate"),
        ("FACE", "Purse lips (hold) -> Energy Shield"),
        ("FACE", "Tilt head -> aim  (left shoulder = up)"),
        ("KEY", "1 Transform  2 Blaster (hold)  3 Scanner  4 Rage"),
        ("KEY", "5 Shield (hold)  6 Ultimate  SPACE charge/fire"),
        ("KEY", "UP/DOWN or W/S aim   ESC pause   R restart"),
        ("KEY", "F fullscreen   M mute   F3 diagnostics   H help"),
    ]
    for i, (kind, s) in enumerate(rows):
        y = 130 + i * 46
        text(surf, kind, (220, y), 26, BLUE if kind == "FACE" else GREEN)
        text(surf, s, (310, y), 26, WHITE)
    text(surf, "Press any key to close", (W // 2, H - 50), 24, GREY, "center")


def draw_diagnostics(surf, meas, frame: GestureFrame | None, state_name: str, fps: float, cal, lines: list[str]) -> None:
    panel(surf, (16, 160, 340, 320), 200, GREEN)
    y = 170
    text(surf, "DIAGNOSTICS (F3)", (28, y), 22, GREEN)
    y += 28
    rows = [f"state: {state_name}", f"fps: {fps:0.1f}"]
    if meas is not None:
        rows += [f"mouth_open: {meas.mouth_open:0.3f}", f"brow_raise: {meas.brow_raise:0.3f}",
                 f"lip_width:  {meas.lip_width:0.3f}", f"roll_deg:   {meas.roll_deg:+0.1f}"]
    else:
        rows.append("measurements: none")
    if frame is not None:
        rows.append("active: " + (", ".join(g.name for g in frame.active) or "-"))
        rows.append("scores: " + " ".join(f"{g.name[:5]}={s:0.2f}" for g, s in frame.scores.items()))
        rows.append(f"aim: {frame.aim:+0.2f}  neutral: {frame.neutral}")
        rows.append(f"ready: {frame.tracking_ready}")
    rows.append(f"calibrated: {cal.calibrated}")
    rows += lines
    for r in rows:
        text(surf, r, (28, y), 18, WHITE)
        y += 22


def draw_landmarks(surf, lm, mapper, color=(90, 255, 160)) -> None:
    """Draw a sparse set of landmark dots for diagnostics."""
    step = 3
    for i in range(0, len(lm), step):
        x, y = mapper.to_screen(float(lm[i][0]), float(lm[i][1]))
        if 0 <= x < W and 0 <= y < H:
            surf.fill(color, (int(x), int(y), 2, 2))


def draw_title(surf, t: float) -> None:
    glow = 0.5 + 0.5 * math.sin(t * 2.2)
    text(surf, "AURA BREAKER", (W // 2, 110), 124, (255, int(190 + 50 * glow), 40), "center")
    text(surf, "SAIYAN AWAKENING", (W // 2, 196), 56, WHITE, "center")
    text(surf, C.TAGLINE, (W // 2, 250), 28, BLUE, "center")
