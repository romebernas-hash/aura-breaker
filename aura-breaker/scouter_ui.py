"""Scouter-style camera panel ("SCOUTER LINK") and the updated controls help screen.

The panel replaces the plain camera box. It shows what the game sees of you:
  * the live webcam feed with a tinted, scan-lined, animated display,
  * face lock-on brackets and the detected expression,
  * your hand skeleton with a pose label (OPEN PALM / FIST),
  * segmented meters for the face inputs and what each one does,
  * hand tracking status and the current action,
  * a KI gauge with the ultimate marker and an ULTIMATE READY indicator,
  * the panel colour follows your form: blue (base), gold (awakened), red (rage), white-gold (ultimate).

It reads everything from the ``App`` object passed to :meth:`ScouterCam.draw` (duck typing, no imports of main).
"""
from __future__ import annotations

import math

import pygame

import config as C
import ui
from battle_system import Ability
from face_controls import Gesture as G
from hand_controls import HandPose, classify_pose

W, H = C.LOGICAL_W, C.LOGICAL_H

HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

X, Y, PW, PH = 12, 152, 304, 460          # panel rectangle (below the player HUD, above the ability bar)
CW, CH = 288, 162                         # camera feed size (16:9)

THEMES = {
    "base": (90, 185, 255),
    "gold": (255, 205, 60),
    "rage": (255, 80, 70),
    "ultimate": (255, 240, 180),
}


def _dim(c, k):
    return tuple(int(v * k) for v in c)


def seg_bar(surf, x, y, w, frac, color, segs=12, h=11, gap=2):
    """Segmented meter."""
    frac = max(0.0, min(1.0, frac))
    sw = (w - gap * (segs - 1)) / segs
    lit = frac * segs
    for i in range(segs):
        rx = int(x + i * (sw + gap))
        on = lit - i
        col = color if on >= 1 else (_dim(color, 0.35 + 0.65 * max(0.0, on)) if on > 0 else (34, 40, 62))
        pygame.draw.rect(surf, col, (rx, y, max(2, int(sw)), h))


def corner_brackets(surf, rect, color, size=10, width=2):
    x, y, w, h = rect
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        pygame.draw.line(surf, color, (px, py), (px + dx * size, py), width)
        pygame.draw.line(surf, color, (px, py), (px, py + dy * size), width)


class ScouterCam:
    def __init__(self) -> None:
        self._panel_cache: dict[tuple, pygame.Surface] = {}
        self._scan = pygame.Surface((CW, CH), pygame.SRCALPHA)
        for yy in range(0, CH, 3):
            pygame.draw.line(self._scan, (0, 0, 0, 55), (0, yy), (CW, yy))
        for i in range(10):  # soft vignette
            pygame.draw.rect(self._scan, (0, 0, 0, 10 + i * 4), (i, i, CW - 2 * i, CH - 2 * i), 1)
        self._tint = pygame.Surface((CW, CH), pygame.SRCALPHA)
        self._tint.fill((0, 50, 80, 46))

    # ------------------------------------------------------------------
    def _panel(self, color) -> pygame.Surface:
        key = tuple(color)
        s = self._panel_cache.get(key)
        if s is None:
            if len(self._panel_cache) > 8:
                self._panel_cache.clear()
            s = pygame.Surface((PW, PH), pygame.SRCALPHA)
            cut = 16
            poly = [(cut, 0), (PW - cut, 0), (PW, cut), (PW, PH - cut), (PW - cut, PH), (cut, PH), (0, PH - cut), (0, cut)]
            pygame.draw.polygon(s, (8, 12, 26, 205), poly)
            pygame.draw.polygon(s, (*color, 255), poly, 2)
            pygame.draw.polygon(s, (*_dim(color, 0.35), 255), [(cut + 4, 4), (PW - cut - 4, 4), (PW - 4, cut + 4), (PW - 4, PH - cut - 4),
                                                                (PW - cut - 4, PH - 4), (cut + 4, PH - 4), (4, PH - cut - 4), (4, cut + 4)], 1)
            pygame.draw.rect(s, (*_dim(color, 0.55), 255), (10, 6, PW - 20, 22))
            self._panel_cache[key] = s
        return s

    @staticmethod
    def _theme(app) -> tuple[tuple[int, int, int], str]:
        b = app.battle
        if b is None:
            return THEMES["base"], "BASE"
        if b.fsm.state.name == "ULTIMATE":
            return THEMES["ultimate"], "ULTIMATE"
        if b.rage_time > 0:
            return THEMES["rage"], f"RAGE  {b.rage_time:0.0f}s"
        if b.transformed:
            return THEMES["gold"], f"AWAKENED  {b.transform_time:0.0f}s"
        return THEMES["base"], "BASE FORM"

    # ------------------------------------------------------------------
    def draw(self, s: pygame.Surface, app) -> None:
        t = app.t
        color, form = self._theme(app)
        s.blit(self._panel(color), (X, Y))
        pipe = app.pipeline
        # header
        ok_link = pipe is not None and getattr(pipe, "status", "") == "running"
        ui.text(s, "SCOUTER LINK", (X + 18, Y + 9), 18, ui.WHITE, shadow=False)
        led = ui.GREEN if (ok_link and app.anchor.present) else (ui.GOLD if ok_link else ui.RED)
        blink = 0.6 + 0.4 * math.sin(t * 6)
        pygame.draw.circle(s, _dim(led, blink), (X + PW - 70, Y + 17), 5)
        ui.text(s, "LIVE" if ok_link else "NO SIGNAL", (X + PW - 18, Y + 9), 16, led, "right", shadow=False)

        # ---- camera feed -------------------------------------------------
        cx, cy = X + 8, Y + 32
        feed = app.draw_camera(s, (CW, CH), (cx, cy))
        if not feed:
            pygame.draw.rect(s, (14, 18, 34), (cx, cy, CW, CH))
            for i in range(-CH, CW, 18):
                pygame.draw.line(s, (26, 32, 58), (cx + i, cy + CH), (cx + i + CH, cy), 6)
            msg = "NO CAMERA" if pipe is None else f"CAMERA: {getattr(pipe, 'status', '?').upper()}"
            ui.text(s, msg, (cx + CW // 2, cy + CH // 2 - 8), 24, ui.RED, "center")
            ui.text(s, "keyboard fallback active", (cx + CW // 2, cy + CH // 2 + 16), 16, ui.GREY, "center")
        else:
            s.blit(self._tint, (cx, cy))
            s.blit(self._scan, (cx, cy))
            sweep = int((t * 70) % CH)
            sw = pygame.Surface((CW, 3), pygame.SRCALPHA)
            sw.fill((*color, 70))
            s.blit(sw, (cx, cy + sweep))
            self._draw_face_lock(s, app, cx, cy, color)
            self._draw_hands(s, app, cx, cy)
            self._draw_action_strip(s, app, cx, cy, color, t)
        corner_brackets(s, (cx, cy, CW, CH), color, 14, 3)

        # ---- face inputs -------------------------------------------------
        y = cy + CH + 10
        ui.text(s, "FACE INPUTS", (X + 14, y), 15, ui.GREY, shadow=False)
        y += 17
        sc = app.gframe.scores if app.gframe else {}
        hands_on = app.hands_active
        rows = [("ANGRY", G.BROW_LOWER, "TRANSFORM", ui.RED), ("SURPRISED", G.BROW_RAISE, "SCANNER", ui.BLUE),
                ("MOUTH", G.MOUTH_OPEN, "KI CHARGE" if hands_on else "BLAST", ui.GOLD)]
        if not hands_on:
            rows.append(("LIPS", G.PURSE, "SHIELD", ui.GREEN))
        for label, g, fn, col in rows:
            v = max(0.0, min(1.0, float(sc.get(g, 0))))
            active = v >= C.GESTURE_ON
            ui.text(s, label, (X + 14, y), 16, col if active else ui.WHITE, shadow=False)
            seg_bar(s, X + 96, y + 3, 110, v, col)
            ui.text(s, fn, (X + PW - 14, y), 14, col if active else ui.GREY, "right", shadow=False)
            y += 22

        # ---- hands -------------------------------------------------------
        y += 6
        ui.text(s, "HAND INPUTS", (X + 14, y), 15, ui.GREY, shadow=False)
        y += 17
        hf = app.hframe
        hs = getattr(pipe, "hand_status", "disabled")
        if pipe is None:
            line, col = "no camera", ui.GREY
        elif hs == "starting":
            line, col = "loading hand model...", ui.GREY
        elif hs == "unavailable":
            line, col = "UNAVAILABLE - keys / face only", ui.RED
        elif hs == "disabled":
            line, col = "off (--no-hands)", ui.GREY
        elif hf is None or hf.hands == 0:
            line, col = "show your hand to the camera", ui.GOLD
        else:
            line, col = f"{hf.pose.value.upper()}   x{hf.hands}", ui.GREEN
        ui.text(s, line, (X + 14, y), 16, col, shadow=False)
        if hf is not None and hf.hands:
            seg_bar(s, X + PW - 14 - 100, y + 3, 100, hf.confidence, ui.GREEN, 10)
        y += 20
        ui.text(s, "OPEN=BLAST  FIST=SHIELD  2 HANDS=ULTIMATE", (X + 14, y), 13, ui.GREY, shadow=False)
        y += 18
        if hf is not None and hf.ultimate_progress > 0:
            seg_bar(s, X + 14, y, PW - 28, hf.ultimate_progress, (255, 240, 180), 20, 9)
        y += 14

        # ---- ki gauge ----------------------------------------------------
        b = app.battle
        if b is not None:
            y += 4
            ui.text(s, "KI", (X + 14, y), 16, ui.WHITE, shadow=False)
            need = C.ULTIMATE_COST
            bx, bw = X + 40, PW - 54
            frac = b.energy / C.PLAYER_MAX_ENERGY
            pulse = 0.75 + 0.25 * math.sin(t * 14) if getattr(b, "ki_charging", False) else 1.0
            kcol = _dim((255, 215, 80) if b.transformed else (90, 170, 255), pulse)
            seg_bar(s, bx, y + 1, bw, frac, kcol, 25, 16, 1)
            nx = int(bx + bw * need / C.PLAYER_MAX_ENERGY)
            pygame.draw.line(s, (255, 255, 255), (nx, y - 3), (nx, y + 21), 2)
            ui.text(s, "ULT", (nx, y + 22), 12, ui.WHITE, "midtop", shadow=False)
            ui.text(s, f"{int(b.energy)}", (X + PW - 14, y - 16), 14, ui.WHITE, "right", shadow=False)
            y += 40
            cd = b.cooldowns[Ability.ULTIMATE]
            if b.energy >= need and cd <= 0:
                a = 0.82 + 0.18 * math.sin(t * 9)
                ui.text(s, "ULTIMATE READY!", (X + PW // 2, y), 22, _dim(ui.GOLD, a), "midtop")
            elif cd > 0:
                ui.text(s, f"ULTIMATE COOLING  {cd:0.0f}s", (X + PW // 2, y + 2), 17, ui.GREY, "midtop", shadow=False)
            else:
                how = "open mouth / C" if hands_on else "hold C"
                ui.text(s, f"ULTIMATE needs {need:.0f} KI - {how}", (X + PW // 2, y + 2), 15, ui.WHITE, "midtop", shadow=False)
            y += 26
        # form chip
        ui.text(s, "FORM", (X + 14, Y + PH - 24), 14, ui.GREY, shadow=False)
        ui.text(s, form, (X + PW - 14, Y + PH - 26), 18, color, "right", shadow=False)

    # ------------------------------------------------------------------
    def _map(self, app, nx, ny, cx, cy):
        lx, ly = app.mapper_fm.to_screen(float(nx), float(ny))
        return cx + lx * CW / W, cy + ly * CH / H

    def _draw_face_lock(self, s, app, cx, cy, color) -> None:
        a = app.anchor
        if not a.present:
            ui.text(s, "FACE LOST", (cx + CW // 2, cy + 12), 18, ui.RED, "center")
            return
        fw, fh = a.face_w * CW / W, a.face_h * CH / H
        hx, hy = cx + a.head[0] * CW / W, cy + a.head[1] * CH / H
        rect = (hx - fw / 2 - 4, hy - fh / 2 - 4, fw + 8, fh + 8)
        corner_brackets(s, rect, color, 8, 2)
        tag = "FACE"
        sc = app.gframe.scores if app.gframe else {}
        if sc.get(G.BROW_LOWER, 0) >= 0.5:
            tag = "ANGRY"
        elif sc.get(G.BROW_RAISE, 0) >= 0.5:
            tag = "SURPRISED"
        elif sc.get(G.MOUTH_OPEN, 0) >= 0.5:
            tag = "KI MOUTH"
        ui.text(s, tag, (rect[0], rect[1] - 14), 13, color, shadow=False)

    def _draw_hands(self, s, app, cx, cy) -> None:
        if not app.hand_obs or app.mapper_fm is None:
            return
        for ho in app.hand_obs:
            pose, _ = classify_pose(ho.lm, app.hand_engine.cfg, app.hand_engine.aspect)
            col = {HandPose.OPEN: ui.GREEN, HandPose.FIST: ui.RED}.get(pose, ui.GOLD)
            pts = [self._map(app, ho.lm[i, 0], ho.lm[i, 1], cx, cy) for i in range(21)]
            for a, b in HAND_BONES:
                pygame.draw.line(s, _dim(col, 0.8), pts[a], pts[b], 2)
            for p in pts:
                pygame.draw.circle(s, col, (int(p[0]), int(p[1])), 3)
            wx, wy = pts[0]
            ui.text(s, pose.value.upper(), (int(max(cx + 4, min(cx + CW - 70, wx - 28))), int(min(cy + CH - 34, wy + 6))), 14, col, shadow=True)

    def _draw_action_strip(self, s, app, cx, cy, color, t) -> None:
        hf = app.hframe
        b = app.battle
        text = ""
        if getattr(b, "ki_charging", False):
            text = "KI CHARGING"
        elif hf is not None and hf.action:
            text = hf.action
        elif hf is not None and hf.message:
            text = hf.message
        if not text:
            return
        strip = pygame.Surface((CW, 22), pygame.SRCALPHA)
        strip.fill((0, 0, 0, 150))
        s.blit(strip, (cx, cy + CH - 22))
        a = 0.7 + 0.3 * math.sin(t * 10)
        ui.text(s, text, (cx + CW // 2, cy + CH - 11), 16, _dim(color if hf is None or hf.action or getattr(b, "ki_charging", False) else ui.GREY, a),
                "center", shadow=False)


# --------------------------------------------------------------------------
# Controls help (replaces ui.draw_controls_help: adds hands and Ki Charge)
# --------------------------------------------------------------------------
def draw_controls_help(surf, t: float) -> None:
    ui.dim_overlay(surf, 215)
    ui.text(surf, "CONTROLS", (W // 2, 44), 56, ui.GOLD, "center")
    rows = [
        ("HANDS", ui.GREEN, "Open palm (hold) -> charge Ki Blast.  Close your hand -> FIRE"),
        ("HANDS", ui.GREEN, "Closed fist (hold) -> Energy Shield"),
        ("HANDS", ui.GREEN, "Both hands together (hold) -> ULTIMATE (needs %d KI)" % C.ULTIMATE_COST),
        ("FACE", ui.BLUE, "Brows down -> Transform / Rage     Brows up -> Scanner"),
        ("FACE", ui.BLUE, "Open mouth (hold) -> KI CHARGE: power up to refill energy"),
        ("FACE", ui.BLUE, "Tilt head -> aim the blast"),
        ("KEYS", ui.GOLD, "1 Transform  3 Scanner  4 Rage  5 Shield  6 Ultimate"),
        ("KEYS", ui.GOLD, "SPACE Ki Blast (hold / release)   C Ki Charge (hold)"),
        ("KEYS", ui.GOLD, "UP / DOWN aim   ESC pause   R restart"),
        ("KEYS", ui.GOLD, "F fullscreen  M mute  F3 diagnostics  F4 camera view  H help"),
    ]
    for i, (kind, col, s) in enumerate(rows):
        y = 120 + i * 44
        ui.text(surf, kind, (150, y), 26, col)
        ui.text(surf, s, (260, y), 26, ui.WHITE)
    ui.text(surf, "Ki Charge refills energy fast, but you cannot act and the enemy keeps attacking!", (W // 2, 588), 24, ui.GOLD, "center")
    ui.text(surf, "Tip: transform first, then power up to 60 KI for the ultimate.", (W // 2, 622), 22, ui.GREY, "center")
    ui.text(surf, "Press any key to close", (W // 2, H - 40), 24, ui.GREY, "center")