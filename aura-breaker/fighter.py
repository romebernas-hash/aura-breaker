"""Side-view player fighter (original Goku-inspired artwork, drawn procedurally).

The fighter always faces RIGHT, toward the enemy, so nothing is ever mirrored and
asymmetric hair / clothing stays correct. Poses are driven by the battle state:

    idle, ki (power-up), charge (open hand + orb), fire (arm thrust along the aim),
    guard (shield), transform, rage, ult_charge (cupped hands at the hip),
    ult_fire (both hands thrust forward), hit reaction, victory, defeat.

``fighter.hand`` is the world position of the firing hand; beams, the charge orb and
the ultimate orb are spawned from it (see ``effects.aim_geometry``).

Only ``pygame.draw`` primitives are used, so a transparent PNG sprite sheet can replace
this later without touching the battle code.
"""
from __future__ import annotations

import math
import random

import pygame

import config as C

DEG = math.pi / 180.0
OUTLINE = (32, 22, 20)
SKIN = (242, 192, 150)
SKIN_SH = (214, 160, 120)
ORANGE = (242, 122, 28)
ORANGE_BACK = (196, 94, 20)
BLUE = (36, 66, 172)
BLUE_BACK = (28, 50, 130)
HAIR_BLACK = (30, 26, 36)
HAIR_GOLD = (255, 214, 64)
HAIR_GOLD_HI = (255, 246, 160)

# pose targets. Hand offsets are relative to the shoulder (+x = toward the enemy, +y = down).
POSES: dict[str, dict] = {
    "idle":      dict(lean=6,   drop=0,  lunge=0,  stf=62, stb=58, tilt=0,   fh=(62, 48),   bh=(36, 70)),
    "ki":        dict(lean=14,  drop=46, lunge=0,  stf=72, stb=72, tilt=-8,  fh=(30, 118),  bh=(0, 120)),
    "charge":    dict(lean=8,   drop=18, lunge=0,  stf=70, stb=62, tilt=0,   fh=None,       bh=None),
    "fire":      dict(lean=16,  drop=22, lunge=28, stf=82, stb=60, tilt=2,   fh=None,       bh=(-14, 62)),
    "guard":     dict(lean=0,   drop=14, lunge=0,  stf=60, stb=60, tilt=0,   fh=(50, -90),  bh=(34, -30)),
    "transform": dict(lean=-6,  drop=32, lunge=0,  stf=78, stb=78, tilt=-14, fh=(78, 22),   bh=(-34, 26)),
    "rage":      dict(lean=18,  drop=28, lunge=0,  stf=74, stb=70, tilt=4,   fh=(42, 98),   bh=(6, 100)),
    "ult_charge": dict(lean=-2, drop=44, lunge=0,  stf=84, stb=78, tilt=0,   fh=(-26, 138), bh=(-50, 134)),
    "ult_fire":  dict(lean=18,  drop=26, lunge=34, stf=90, stb=66, tilt=2,   fh=None,       bh=None),
    "victory":   dict(lean=-4,  drop=0,  lunge=0,  stf=60, stb=56, tilt=-6,  fh=(34, -124), bh=(30, 72)),
    "defeat":    dict(lean=52,  drop=96, lunge=-14, stf=34, stb=18, tilt=20, fh=(44, 112),  bh=(12, 118)),
}
NUM_KEYS = ("lean", "drop", "lunge", "stf", "stb", "tilt")


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ik2(a, b, l1: float, l2: float, prefer=(0.0, 1.0)):
    """Two-bone IK. Returns (joint, end). The joint is on the side facing ``prefer``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(dx, dy) or 1e-6
    reach = min(dist, l1 + l2 - 0.5)
    ux, uy = dx / dist, dy / dist
    end = (a[0] + ux * reach, a[1] + uy * reach)
    along = (l1 * l1 - l2 * l2 + reach * reach) / (2 * reach)
    h = math.sqrt(max(0.0, l1 * l1 - along * along))
    mx, my = a[0] + ux * along, a[1] + uy * along
    c1 = (mx - uy * h, my + ux * h)
    c2 = (mx + uy * h, my - ux * h)
    d1 = (c1[0] - mx) * prefer[0] + (c1[1] - my) * prefer[1]
    d2 = (c2[0] - mx) * prefer[0] + (c2[1] - my) * prefer[1]
    return (c1 if d1 >= d2 else c2), end


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def limb(surf, pts, width, color, outline=OUTLINE, ow=4):
    """Rounded multi-segment limb with an outline."""
    for col, w in ((outline, width + ow), (color, width)):
        for p, q in zip(pts, pts[1:]):
            pygame.draw.line(surf, col, p, q, int(w))
        for p in pts:
            pygame.draw.circle(surf, col, (int(p[0]), int(p[1])), int(w / 2))


class FighterRenderer:
    def __init__(self) -> None:
        self.t = 0.0
        self.pose = "idle"
        self.vals = {k: float(POSES["idle"][k]) for k in NUM_KEYS}
        self.fh = list(map(float, POSES["idle"]["fh"]))
        self.bh = list(map(float, POSES["idle"]["bh"]))
        self.hit_t = 0.0
        self.hand: tuple[float, float] = (420.0, 340.0)
        self.joints: dict = {}
        self.rng = random.Random(7)
        self.hair_mix = 0.0
        self.tint: str | None = None
        self.expr = dict(brow_up=0.0, brow_down=0.0, mouth=0.0, purse=0.0)
        self._sh = (320.0, 300.0)
        self._open_hand = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.pose = "idle"
        self.hit_t = 0.0
        self.vals = {k: float(POSES["idle"][k]) for k in NUM_KEYS}
        self.fh = list(map(float, POSES["idle"]["fh"]))
        self.bh = list(map(float, POSES["idle"]["bh"]))

    def on_event(self, ev) -> None:
        if ev.kind == "player_hit" and ev.data.get("damage", 0) > 0:
            self.hit_t = 0.32
        elif ev.kind == "shield_hit":
            self.hit_t = max(self.hit_t, 0.10)

    # ------------------------------------------------------------------
    def _pose_key(self, b) -> str:
        if b is None:
            return "idle"
        st = b.fsm.state.name
        if st == "DEFEAT":
            return "defeat"
        if st == "VICTORY":
            return "victory"
        if st == "ULTIMATE":
            return "ult_charge" if b.ult_time < C.ULTIMATE_CHARGE_TIME else "ult_fire"
        if st == "TRANSFORMING":
            return "transform"
        if st == "RAGE_BURST":
            return "rage"
        if st == "ATTACKING":
            return "fire"
        if st == "CHARGING":
            return "charge"
        if st == "DEFENDING":
            return "guard"
        if getattr(b, "ki_charging", False):
            return "ki"
        return "idle"

    def update(self, dt: float, b, anchor, enemy_pos, expr: dict | None = None, hair_mix: float = 0.0) -> None:
        dt = max(0.0, min(dt, 0.05))
        self.t += dt
        self.hit_t = max(0.0, self.hit_t - dt)
        self.hair_mix += (hair_mix - self.hair_mix) * (1 - math.exp(-dt * 8))
        if expr:
            self.expr.update(expr)
        key = self._pose_key(b)
        snap = key != self.pose and key in ("fire", "ult_fire")
        self.pose = key
        pz = POSES[key]
        k = anchor.face_h / 170.0
        rate = 1.0 if snap else 1 - math.exp(-dt * 15)
        for n in NUM_KEYS:
            self.vals[n] = lerp(self.vals[n], float(pz[n]), rate)
        v = dict(self.vals)
        hit = self.hit_t / 0.32 if self.hit_t > 0 else 0.0
        v["lean"] -= 24 * hit
        v["lunge"] -= 22 * hit
        v["tilt"] -= 10 * hit
        bob = math.sin(self.t * 2.4) * 4 * k if key == "idle" else math.sin(self.t * 5) * 2 * k
        shake = {"ki": 2.5, "transform": 4.0, "rage": 3.0, "ult_charge": 2.0}.get(key, 0.0)
        jx = self.rng.uniform(-1, 1) * shake * k
        jy = self.rng.uniform(-1, 1) * shake * k * 0.5

        ground = anchor.head[1] + 367 * k
        px = anchor.head[0] + v["lunge"] * k + jx
        pelvis = (px, ground - (185 - v["drop"]) * k + bob + jy)
        lean = v["lean"] * DEG
        sh = (pelvis[0] + 115 * k * math.sin(lean), pelvis[1] - 115 * k * math.cos(lean))
        self._sh = sh
        tilt = lean + v["tilt"] * DEG
        head = (sh[0] + 50 * k * math.sin(tilt), sh[1] - 50 * k * math.cos(tilt))

        # hand targets (relative to the shoulder)
        ang = self._aim_angle(b, enemy_pos, sh)
        dirv = (math.cos(ang), math.sin(ang))
        if key == "charge":
            grow = b.charge if b is not None else 0.0
            fh = (dirv[0] * (92 + 16 * grow), dirv[1] * (92 + 16 * grow))
            bh = (fh[0] * 0.62 - 6, fh[1] * 0.62 + 16)
        elif key == "fire":
            fh = (dirv[0] * 136, dirv[1] * 136)
            bh = pz["bh"]
        elif key == "ult_fire":
            fh = (dirv[0] * 134, dirv[1] * 134)
            bh = (dirv[0] * 116 - 2, dirv[1] * 116 + 12)
        else:
            fh, bh = pz["fh"], pz["bh"]
        if key in ("ki", "transform", "rage", "ult_charge"):
            fh = (fh[0] + self.rng.uniform(-3, 3) * shake / 2, fh[1] + self.rng.uniform(-3, 3) * shake / 2)
        rate_h = 1.0 if snap else 1 - math.exp(-dt * 20)
        self.fh = [lerp(self.fh[0], fh[0], rate_h), lerp(self.fh[1], fh[1], rate_h)]
        self.bh = [lerp(self.bh[0], bh[0], rate_h), lerp(self.bh[1], bh[1], rate_h)]

        sf = (sh[0] + 8 * k, sh[1] + 4 * k)
        sb = (sh[0] - 6 * k, sh[1] + 2 * k)
        hf = (sf[0] + self.fh[0] * k, sf[1] + self.fh[1] * k)
        hb = (sb[0] + self.bh[0] * k, sb[1] + self.bh[1] * k)
        prefer = (-1.0, 0.6) if key in ("ult_charge", "ki", "defeat") else (0.0, 1.0)
        ef, hf = ik2(sf, hf, 72 * k, 68 * k, prefer)
        eb, hb = ik2(sb, hb, 72 * k, 68 * k, prefer)

        ff = (px + v["stf"] * k + v["lunge"] * 0.3 * k, ground)
        fb = (px - v["stb"] * k, ground)
        hip_f = (pelvis[0] + 8 * k, pelvis[1])
        hip_b = (pelvis[0] - 8 * k, pelvis[1])
        kf, ff = ik2(hip_f, ff, 100 * k, 100 * k, (1.0, 0.0))
        kb, fb = ik2(hip_b, fb, 100 * k, 100 * k, (1.0, 0.0))

        if key in ("ult_charge",):
            self.hand = ((hf[0] + hb[0]) / 2, (hf[1] + hb[1]) / 2)
        elif key == "ult_fire":
            self.hand = ((hf[0] + hb[0]) / 2, (hf[1] + hb[1]) / 2)
        else:
            self.hand = hf
        self._open_hand = key in ("charge", "fire")
        self.tint = "red" if key == "rage" or (b is not None and b.rage_time > 0) else None
        self.joints = dict(k=k, ground=ground, pelvis=pelvis, sh=sh, head=head, tilt=tilt, lean=lean, sf=sf, sb=sb,
                           ef=ef, eb=eb, hf=hf, hb=hb, hip_f=hip_f, hip_b=hip_b, kf=kf, kb=kb, ff=ff, fb=fb, key=key)

    def _aim_angle(self, b, enemy_pos, sh) -> float:
        base = math.atan2(enemy_pos[1] - sh[1], enemy_pos[0] - sh[0])
        aim = b.aim if b is not None else 0.0
        return base - math.radians(C.MAX_AIM_DEG) * max(-1.0, min(1.0, aim))

    # ------------------------------------------------------------------
    def draw(self, surf) -> None:
        j = self.joints
        if not j:
            return
        k, key = j["k"], j["key"]
        gold = self.hair_mix
        red = self.tint == "red"
        orange = mix(ORANGE, (255, 90, 60), 0.45) if red else ORANGE
        orange_b = mix(ORANGE_BACK, (190, 50, 40), 0.45) if red else ORANGE_BACK
        # ground shadow
        gx = j["pelvis"][0]
        pygame.draw.ellipse(surf, (10, 10, 16), (int(gx - 95 * k), int(j["ground"] - 12 * k), int(190 * k), int(26 * k)))

        def boot(foot, col):
            pygame.draw.ellipse(surf, OUTLINE, (int(foot[0] - 20 * k), int(foot[1] - 24 * k), int(66 * k), int(30 * k)))
            pygame.draw.ellipse(surf, col, (int(foot[0] - 17 * k), int(foot[1] - 21 * k), int(60 * k), int(24 * k)))

        def arm(sh, el, hd, shade, open_hand):
            limb(surf, [sh, el], 26 * k, orange_b if shade else orange)
            limb(surf, [el, hd], 20 * k, SKIN_SH if shade else SKIN)
            seg = (hd[0] - el[0], hd[1] - el[1])
            n = math.hypot(*seg) or 1.0
            wb = (hd[0] - seg[0] / n * 16 * k, hd[1] - seg[1] / n * 16 * k)
            limb(surf, [wb, (hd[0] - seg[0] / n * 6 * k, hd[1] - seg[1] / n * 6 * k)], 23 * k, BLUE_BACK if shade else BLUE)
            r = int((15 if open_hand else 13) * k)
            pygame.draw.circle(surf, OUTLINE, (int(hd[0]), int(hd[1])), r + 3)
            pygame.draw.circle(surf, SKIN_SH if shade else SKIN, (int(hd[0]), int(hd[1])), r)
            if open_hand:  # fingers fanned along the pointing direction
                for a in (-0.5, -0.17, 0.17, 0.5):
                    ca, sa = math.cos(math.atan2(seg[1], seg[0]) + a), math.sin(math.atan2(seg[1], seg[0]) + a)
                    pygame.draw.line(surf, OUTLINE, (hd[0], hd[1]), (hd[0] + ca * 22 * k, hd[1] + sa * 22 * k), 6)
                    pygame.draw.line(surf, SKIN, (hd[0], hd[1]), (hd[0] + ca * 20 * k, hd[1] + sa * 20 * k), 3)

        # back limbs
        limb(surf, [j["hip_b"], j["kb"], j["fb"]], 32 * k, orange_b)
        boot(j["fb"], BLUE_BACK)
        arm(j["sb"], j["eb"], j["hb"], True, False)
        # torso
        lean = j["lean"]
        ax, ay = math.sin(lean), -math.cos(lean)
        px_, py_ = math.cos(lean), math.sin(lean)
        P, S = j["pelvis"], j["sh"]
        quad = [(P[0] - px_ * 28 * k, P[1] - py_ * 28 * k), (P[0] + px_ * 30 * k, P[1] + py_ * 30 * k),
                (S[0] + px_ * 36 * k, S[1] + py_ * 36 * k), (S[0] - px_ * 34 * k, S[1] - py_ * 34 * k)]
        pygame.draw.polygon(surf, OUTLINE, quad, 0)
        inner = [(P[0] - px_ * 24 * k, P[1] - py_ * 24 * k), (P[0] + px_ * 26 * k, P[1] + py_ * 26 * k),
                 (S[0] + px_ * 32 * k, S[1] + py_ * 32 * k), (S[0] - px_ * 30 * k, S[1] - py_ * 30 * k)]
        pygame.draw.polygon(surf, orange, inner)
        # blue undershirt at the neck and belt sash
        pygame.draw.polygon(surf, BLUE, [(S[0] - px_ * 20 * k, S[1] - py_ * 20 * k), (S[0] + px_ * 24 * k, S[1] + py_ * 24 * k),
                                         (S[0] + px_ * 14 * k - ax * 28 * k, S[1] + py_ * 14 * k - ay * 28 * k),
                                         (S[0] - px_ * 12 * k - ax * 16 * k, S[1] - py_ * 12 * k - ay * 16 * k)])
        b0 = (P[0] + ax * 4 * k, P[1] + ay * 4 * k)
        b1 = (P[0] + ax * 26 * k, P[1] + ay * 26 * k)
        pygame.draw.polygon(surf, BLUE, [(b0[0] - px_ * 28 * k, b0[1] - py_ * 28 * k), (b0[0] + px_ * 30 * k, b0[1] + py_ * 30 * k),
                                         (b1[0] + px_ * 30 * k, b1[1] + py_ * 30 * k), (b1[0] - px_ * 28 * k, b1[1] - py_ * 28 * k)])
        pygame.draw.line(surf, (255, 190, 90), (S[0] - px_ * 24 * k, S[1] - py_ * 24 * k), (P[0] - px_ * 20 * k, P[1] - py_ * 20 * k), 3)
        # front leg
        limb(surf, [j["hip_f"], j["kf"], j["ff"]], 34 * k, orange)
        boot(j["ff"], BLUE)
        # neck + head
        H, Sx = j["head"], j["sh"]
        limb(surf, [(Sx[0] + ax * 4 * k, Sx[1] + ay * 4 * k), H], 20 * k, SKIN_SH)
        self._draw_head(surf, H, j["tilt"], k, gold)
        # front arm last (in front of the body)
        arm(j["sf"], j["ef"], j["hf"], False, self._open_hand)

    def _draw_head(self, surf, H, tilt, k, gold) -> None:
        ca, sa = math.cos(tilt), math.sin(tilt)

        def T(x, y):
            return (H[0] + (x * ca - y * sa) * k, H[1] + (x * sa + y * ca) * k)

        hair = mix(HAIR_BLACK, HAIR_GOLD, gold)
        hi = mix((70, 64, 86), HAIR_GOLD_HI, gold)
        spikes_back = [(-34, 10), (-60, 0), (-44, -18), (-74, -38), (-38, -42), (-58, -74), (-20, -58), (-24, -98),
                       (6, -64), (16, -104), (32, -62), (54, -86), (52, -50), (78, -54), (48, -30), (40, -22), (-30, -20)]
        pts = [T(x, y) for x, y in spikes_back]
        pygame.draw.polygon(surf, OUTLINE, pts)
        # ear + face
        face = [T(x, y) for x, y in ((-30, -20), (36, -26), (48, -2), (44, 18), (30, 40), (4, 48), (-22, 30), (-34, 6))]
        pygame.draw.polygon(surf, OUTLINE, face)
        inner = [T(x, y) for x, y in ((-27, -18), (34, -23), (44, -2), (40, 16), (28, 37), (4, 44), (-19, 28), (-30, 6))]
        pygame.draw.polygon(surf, SKIN, inner)
        ear = T(-18, 4)
        pygame.draw.circle(surf, OUTLINE, (int(ear[0]), int(ear[1])), int(11 * k))
        pygame.draw.circle(surf, SKIN_SH, (int(ear[0]), int(ear[1])), int(8 * k))
        # hair fill (drawn after face so the fringe overlaps the forehead)
        hair_pts = [T(x, y) for x, y in ((-32, 8), (-58, -2), (-42, -19), (-72, -39), (-37, -42), (-56, -72), (-19, -57),
                                         (-22, -95), (6, -63), (16, -101), (31, -61), (52, -84), (50, -49), (75, -52),
                                         (46, -29), (36, -20), (22, -30), (6, -24), (-6, -34), (-20, -22), (-30, -12))]
        pygame.draw.polygon(surf, hair, hair_pts)
        pygame.draw.line(surf, hi, T(-18, -50), T(-12, -80), max(2, int(5 * k)))
        pygame.draw.line(surf, hi, T(18, -50), T(26, -76), max(2, int(5 * k)))
        # nose
        pygame.draw.polygon(surf, OUTLINE, [T(40, -4), T(54, 10), T(40, 14)])
        pygame.draw.polygon(surf, SKIN, [T(39, -2), T(50, 10), T(39, 12)])
        # eye + brow (expression driven by the player's real face)
        e = self.expr
        angry = max(e["brow_down"], 0.55 if self.pose in ("transform", "rage", "ult_charge", "ki") else 0.0)
        eye = T(22, -4)
        pygame.draw.ellipse(surf, OUTLINE, (eye[0] - 11 * k, eye[1] - 8 * k, 24 * k, 17 * k))
        pygame.draw.ellipse(surf, (255, 255, 255), (eye[0] - 9 * k, eye[1] - 6 * k, 20 * k, 13 * k))
        pupil = T(28, -3)
        pygame.draw.circle(surf, (20, 70, 120) if gold > 0.5 else (30, 24, 24), (int(pupil[0]), int(pupil[1])), max(2, int(4.5 * k)))
        up = e["brow_up"]
        b_back = T(8, -20 - 7 * up + 1 * angry)
        b_front = T(36, -20 - 7 * up + 9 * angry)
        pygame.draw.line(surf, OUTLINE, b_back, b_front, max(3, int(6 * k)))
        # mouth
        m = e["mouth"]
        mc = T(30, 28)
        if self.pose in ("ki", "transform", "rage") and m < 0.3:
            m = 0.7
        if m > 0.1:
            pygame.draw.ellipse(surf, (90, 20, 30), (mc[0] - 9 * k, mc[1] - 3 * k, 18 * k, (6 + 16 * min(1.0, m)) * k))
        elif e["purse"] > 0.3:
            pygame.draw.ellipse(surf, (170, 70, 80), (mc[0] - 4 * k, mc[1], 8 * k, 6 * k))
        else:
            pygame.draw.line(surf, (150, 70, 70), T(18, 29), T(36, 31), 3)