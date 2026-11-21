#!/usr/bin/env python3
"""AURA BREAKER: SAIYAN AWAKENING - face-controlled anime battle game.

Run:  python main.py                (face + hands + keyboard fallback)
      python main.py --face-only    (body controls only, keyboard combat keys disabled)
      python main.py --no-hands     (face + keyboard only, old mouth/lip controls)
      python main.py --help         (options: --no-camera, --demo-face, --diagnostic, --selftest)

In game:  F4 = toggle small face box / full-screen webcam background
          F3 = diagnostics overlay
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AURA BREAKER: SAIYAN AWAKENING")
    ap.add_argument("--no-camera", action="store_true", help="keyboard only; do not open the webcam")
    ap.add_argument("--demo-face", action="store_true", help="scripted simulated face instead of a webcam")
    ap.add_argument("--face-only", action="store_true", help="disable keyboard combat controls; face/hands only")
    ap.add_argument("--no-hands", action="store_true", help="disable hand tracking (face + keyboard only)")
    ap.add_argument("--diagnostic", action="store_true", help="start with the diagnostics overlay")
    ap.add_argument("--camera", type=int, default=None, help="camera index override")
    ap.add_argument("--quality", choices=["low", "standard", "high"], help="effects quality override")
    ap.add_argument("--headless", action="store_true", help="dummy video/audio drivers")
    ap.add_argument("--selftest", type=float, metavar="SECONDS", help="run a scripted automatic playthrough and exit")
    ap.add_argument("--shots", type=str, help="with --selftest: directory for screenshots")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else parse_args([])
if ARGS.headless or ARGS.selftest:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

import arena  # noqa: E402
import config as C  # noqa: E402
import enemy_art  # noqa: E402
import scouter_ui  # noqa: E402
import sound_design  # noqa: E402
import ui  # noqa: E402
from animation import AvatarRenderer, EnemyRenderer  # noqa: E402
from audio_manager import AudioManager  # noqa: E402
from battle_ki import KiBattle  # noqa: E402  (also sets the new ultimate cost)
from battle_system import Ability, Battle, Event  # noqa: E402
from camera import FrameMapper  # noqa: E402
from effects import Fx, FxContext, HairOverlay  # noqa: E402
from face_controls import (  # noqa: E402
    Calibration, Calibrator, FaceInputMapper, FaceMeasurements, Gesture, GestureDetector, GestureFrame,
)
from fighter import FighterRenderer  # noqa: E402
from face_tracking import (  # noqa: E402
    AnchorSmoother, PlayerAnchor, SyntheticFace, TrackingPipeline, compute_anchor, neutral_measurements,
)
from hand_controls import HandFrame, HandGestureEngine, HandObs, HandPose  # noqa: E402
from game_state import COMBAT_STATES, Commands, GameState as S, StateMachine  # noqa: E402

log = logging.getLogger("aura")
W, H = C.LOGICAL_W, C.LOGICAL_H

QUALITY_NAMES = {"low": "LOW", "standard": "STANDARD", "high": "HIGH"}

# where the side-view fighter stands (the enemy stands at C.ENEMY_POS on the right)
STAGE_HEAD = (440.0, 248.0)
STAGE_FACE = (130.0, 170.0)   # virtual face width / height that size the aura, shield and fighter


def demo_face_script(t: float) -> FaceMeasurements:
    """Looping scripted face used by --demo-face (exercises the real gesture pipeline)."""
    d = C.DEFAULT_DELTAS
    m = neutral_measurements()
    seq = [  # (duration, overrides)
        (2.5, {}), (0.7, {"brow_raise": m.brow_raise + d["BROW_LOWER"] * 1.3}), (4.0, {}),
        (1.3, {"mouth_open": m.mouth_open + d["MOUTH_OPEN"] * 1.3}), (1.5, {}),
        (0.7, {"brow_raise": m.brow_raise + d["BROW_RAISE"] * 1.3}), (1.2, {}),
        (1.6, {"lip_width": m.lip_width + d["PURSE"] * 1.4}), (1.2, {}),
        (1.0, {"mouth_open": m.mouth_open + d["MOUTH_OPEN"] * 1.3, "brow_raise": m.brow_raise + d["BROW_RAISE"] * 1.3}),
        (3.0, {"roll_deg": 10.0}), (1.0, {}),
    ]
    total = sum(s[0] for s in seq)
    t = t % total
    for dur, ov in seq:
        if t < dur:
            vals = dict(C.DEFAULT_BASELINE)
            vals.update({k: v for k, v in ov.items()})
            return FaceMeasurements(vals["mouth_open"], vals["brow_raise"], vals["lip_width"], vals["roll_deg"])
        t -= dur
    return m


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        try:
            from scripts import setup_assets
            setup_assets.ensure_generated_assets()
        except Exception as exc:  # asset generation must never stop the game from starting
            log.error("Could not generate assets: %s", exc)
        sound_design.ensure_sound_pack()   # original energy sound pack + new music (generated once, never fatal)

        pygame.init()
        pygame.display.set_caption(C.TITLE)
        self.settings = C.Settings.load()
        if args.quality:
            self.settings.quality = args.quality
        if args.camera is not None:
            self.settings.camera_index = args.camera
        self.screen = pygame.Surface((W, H)).convert() if False else pygame.Surface((W, H))
        self.world = pygame.Surface((W, H))
        self.window = None
        self._apply_display()

        self.audio = AudioManager(self.settings)
        self.fsm = StateMachine(S.MENU)
        self.battle: Battle | None = None
        self.fx = Fx(self.settings.quality)
        self.enemy = EnemyRenderer()
        if self.enemy.sprite is None:        # no user sprite: use the Frieza-inspired villain, facing the player
            self.enemy.base = enemy_art.build_tyrant_body()
            self.enemy.NAME = "VOID EMPEROR"
        self.fighter = FighterRenderer()     # side-view player, always facing the enemy
        self.arena = arena.RockyArena()
        self.scouter = scouter_ui.ScouterCam()
        self.stage = PlayerAnchor.from_params(STAGE_HEAD, STAGE_FACE[0], STAGE_FACE[1], 0.0, True)
        self.avatar = AvatarRenderer()
        self.hair = HairOverlay()
        self.bg = ui.ArenaBackground()
        self.banner = ui.Banner()
        self.toasts = ui.Toasts()
        self.scanner = ui.ScannerReadout()
        self.t = 0.0
        self.fps = 0.0
        self.running = True
        self.show_diag = args.diagnostic
        self.show_help = False
        self.held: set[int] = set()
        self.kb = Commands()           # edge commands from the keyboard (cleared each frame)
        self.key_aim = 0.0
        self.used_abilities: set[Ability] = set()
        self.event_counts: dict[str, int] = {}
        self.pending_cam_restart: float | None = None

        # face-first display / control options
        self.pip_mode = True                 # True = arena + small face box, False = full-screen webcam background
        self.face_only = args.face_only      # True = keyboard combat keys disabled

        # hand control (primary input for Ki Blast, shield and ultimate when the hand tracker is running)
        self.hand_engine = HandGestureEngine()
        self.hframe: HandFrame | None = None
        self.hand_obs: list[HandObs] = []
        self.hand_screen: tuple[float, float] | None = None   # logical-screen muzzle position
        self._hand_seq = -1
        self._hand_seen = 0.0

        # face control
        self.calibration = Calibration.from_dict(self.settings.calibration)
        self.detector = GestureDetector(self.calibration, self.settings.sensitivity, self.settings.hold_ms / 1000,
                                        self.settings.invert_tilt)
        self.input_mapper = FaceInputMapper()
        self.anchor_smoother = AnchorSmoother()
        self.anchor = PlayerAnchor.default()
        self.gframe: GestureFrame | None = None
        self.track = None
        self.pipeline = None
        self._cam_seq = -1
        self._cam_src: pygame.Surface | None = None
        self.mapper_fm: FrameMapper | None = None
        self.calibrator: Calibrator | None = None
        self.calib_detector: GestureDetector | None = None
        self.calib_return = S.MENU

        # menus
        self.main_menu = ui.Menu(["BATTLE MODE", "TRAINING", "CALIBRATE FACE", "SETTINGS", "CONTROLS", "QUIT"])
        self.pause_menu = ui.Menu(["RESUME", "RESTART BATTLE", "MAIN MENU"], top=320)
        self.result_menu = ui.Menu(["NEXT ROUND", "RESTART", "MAIN MENU"], top=360)
        self.settings_screen = ui.SettingsScreen(self._build_settings_rows())

        self.script: list[tuple[float, str]] = []
        self.script_i = 0
        self.shots_dir: Path | None = None
        self.start_camera()
        self.audio.play_music("music_menu")
        if args.selftest:
            self._setup_selftest()
        elif not self.settings.first_run_done and self.pipeline is not None and not args.demo_face:
            self.begin_calibration(S.MENU)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _apply_display(self) -> None:
        if self.settings.fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            try:
                dw, dh = pygame.display.get_desktop_sizes()[0]
            except (pygame.error, IndexError):
                dw, dh = 1920, 1080
            scale = min(1.0, (dw * 0.92) / W, (dh * 0.85) / H)
            self.window = pygame.display.set_mode((int(W * scale), int(H * scale)), pygame.RESIZABLE)

    def toggle_fullscreen(self) -> None:
        self.settings.fullscreen = not self.settings.fullscreen
        try:
            self._apply_display()
        except pygame.error as exc:
            log.error("Fullscreen toggle failed: %s", exc)
            self.settings.fullscreen = not self.settings.fullscreen

    def _view(self):
        win = pygame.display.get_surface()
        ww, wh = win.get_size()
        scale = min(ww / W, wh / H)
        nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
        return win, scale, ((ww - nw) // 2, (wh - nh) // 2), (nw, nh)

    def to_logical(self, pos) -> tuple[float, float]:
        _, scale, (ox, oy), _ = self._view()
        return ((pos[0] - ox) / scale, (pos[1] - oy) / scale)

    def present(self) -> None:
        win, scale, off, size = self._view()
        if size == (W, H) and off == (0, 0):
            win.blit(self.screen, (0, 0))
        else:
            win.fill((0, 0, 0))
            win.blit(pygame.transform.scale(self.screen, size), off)
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def start_camera(self) -> None:
        if self.args.no_camera:
            self.pipeline = None
            return
        if self.args.demo_face:
            self.pipeline = SyntheticFace(demo_face_script)
        else:
            self.pipeline = TrackingPipeline(self.settings.camera_index, self.settings.mirror,
                                             use_hands=not self.args.no_hands)
        self.pipeline.start()
        status = getattr(self.pipeline, "status", "")
        if status in ("error", "failed", "no_camera"):
            self.toasts.add("CAMERA FAILED - run: python scripts/check_environment.py", ui.RED, 5.0)

    def restart_camera(self) -> None:
        old, self.pipeline = self.pipeline, None
        if old is not None:  # stop in the background so the UI never blocks
            threading.Thread(target=old.stop, daemon=True).start()
        self._cam_seq, self._cam_src, self.mapper_fm = -1, None, None
        self.detector.reset()
        self.start_camera()

    def apply_face_settings(self) -> None:
        self.detector.sensitivity = self.settings.sensitivity
        self.detector.hold_time = self.settings.hold_ms / 1000
        self.detector.invert_tilt = self.settings.invert_tilt

    # ------------------------------------------------------------------
    # Settings rows
    # ------------------------------------------------------------------
    def _build_settings_rows(self) -> list[ui.SettingRow]:
        s = self.settings
        R = ui.SettingRow
        pct = lambda v: f"{int(round(v * 100))}%"  # noqa: E731

        def cycle_quality(d):
            order = C.QUALITY_ORDER
            s.quality = order[(order.index(s.quality) + d) % len(order)]
            self.fx.set_quality(s.quality)

        def adj(attr_get, attr_set, d, step=0.1, lo=0.0, hi=1.0):
            attr_set(round(min(hi, max(lo, attr_get() + d * step)), 3))
            self.audio.apply_settings()

        def vol_row(label, ch):
            return R(label, lambda: pct(s.volumes[ch]),
                     lambda d: adj(lambda: s.volumes[ch], lambda v: s.volumes.__setitem__(ch, v), d))

        def toggle(attr):
            def f(_d):
                setattr(s, attr, not getattr(s, attr))
                if attr == "mirror":
                    self.restart_camera()
                self.apply_face_settings()
            return f

        def change_cam(d):
            s.camera_index = (s.camera_index + d) % 4
            self.pending_cam_restart = 0.9

        def brow(d):
            order = ["auto", "transform", "rage"]
            s.brow_action = order[(order.index(s.brow_action) + d) % 3]

        def sens(d):
            s.sensitivity = round(min(2.0, max(0.4, s.sensitivity + 0.1 * d)), 2)
            self.apply_face_settings()

        def hold(d):
            s.hold_ms = int(min(400, max(0, s.hold_ms + 20 * d)))
            self.apply_face_settings()

        def cam_status():
            if self.pipeline is None:
                return "OFF"
            return f"#{s.camera_index}  {self.pipeline.status.upper()}"

        def mute(_d):
            self.audio.toggle_mute()

        return [
            R("Effects quality", lambda: QUALITY_NAMES[s.quality], cycle_quality),
            R("Fullscreen (F)", lambda: "ON" if s.fullscreen else "OFF", lambda d: self.toggle_fullscreen()),
            R("Mute (M)", lambda: "MUTED" if s.muted else "ON", mute),
            R("Master volume", lambda: pct(s.master_volume),
              lambda d: adj(lambda: s.master_volume, lambda v: setattr(s, "master_volume", v), d)),
            vol_row("Background music", "music"), vol_row("Transformation sounds", "transform"),
            vol_row("Charging", "charge"), vol_row("Energy beams", "beam"), vol_row("Enemy impacts", "impact"),
            vol_row("Shield", "shield"), vol_row("Victory / defeat", "result"),
            R("Camera index", cam_status, change_cam),
            R("Mirror camera", lambda: "ON" if s.mirror else "OFF", toggle("mirror")),
            R("Show face landmarks", lambda: "ON" if s.show_landmarks else "OFF", toggle("show_landmarks")),
            R("Face controls", lambda: "ON" if s.face_controls else "KEYBOARD ONLY", toggle("face_controls")),
            R("Face sensitivity", lambda: f"{s.sensitivity:.1f}x", sens),
            R("Gesture hold time", lambda: f"{s.hold_ms} ms", hold),
            R("Invert tilt aim", lambda: "ON" if s.invert_tilt else "OFF", toggle("invert_tilt")),
            R("Brow gesture action", lambda: s.brow_action.upper(), brow),
            R("Recalibrate face...", lambda: "ENTER", lambda d: None, lambda: self.begin_calibration(S.SETTINGS)),
            R("Retry camera", lambda: "ENTER", lambda d: None, self.restart_camera),
            R("Back", lambda: "ENTER", lambda d: None, lambda: self.goto(S.MENU)),
        ]

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def state(self) -> S:
        return self.fsm.state

    def goto(self, new: S) -> None:
        if self.fsm.state == S.SETTINGS and new == S.MENU:
            self.settings.save()
        if not self.fsm.transition(new):
            self.fsm.force(new)
        if new == S.MENU:
            self.battle = None
            self.fx.reset()
            self.audio.stop_all_effects()
            self.audio.play_music("music_menu")
            self.main_menu.reset()
        elif new == S.SETTINGS:
            self.settings_screen.selected = 0
            self.settings_screen.top = 0

    def start_game(self, training: bool) -> None:
        target = S.TRAINING if training else S.BATTLE_IDLE
        if not self.fsm.transition(target):
            self.fsm.force(target)
        self.battle = KiBattle(fsm=self.fsm, training=training)
        self.fx.reset()
        self.enemy.reset()
        self.used_abilities.clear()
        self.detector.reset()
        self.input_mapper.reset()
        self.hand_engine.reset()
        self.fighter.reset()
        self.key_aim = 0.0
        self.banner.t = 99.0
        self.toasts.items.clear()
        self.audio.play_music("music")
        self.toasts.add("ROUND 1 - FIGHT!" if not training else "TRAINING - TRY EVERY ABILITY", ui.GOLD, 2.2)
        self.toasts.add("KI CHARGE: HOLD C  OR  OPEN YOUR MOUTH", ui.BLUE, 3.5)
        if self.face_only:
            self.toasts.add("FACE-ONLY MODE: use your face!", ui.GREEN, 2.5)

    def restart_battle(self) -> None:
        if self.battle is None:
            return
        self.battle.restart()
        self.fx.reset()
        self.enemy.reset()
        self.used_abilities.clear()
        self.hand_engine.reset()
        self.fighter.reset()
        self.audio.stop_all_effects()
        self.toasts.add("RESTARTED", ui.GOLD)

    def begin_calibration(self, ret: S) -> None:
        self.calib_return = ret
        if not self.fsm.transition(S.CALIBRATION):
            self.fsm.force(S.CALIBRATION)
        self.calibrator = Calibrator()
        self.calib_detector = None

    def end_calibration(self, save: bool) -> None:
        if save and self.calibrator is not None and self.calibrator.finished:
            self.calibration = self.calibrator.build()
            self.settings.calibration = self.calibration.to_dict()
            self.detector.set_calibration(self.calibration)
            self.input_mapper.reset()
            self.toasts.add("CALIBRATION SAVED", ui.GREEN)
        self.settings.first_run_done = True
        self.settings.save()
        self.calibrator = None
        self.calib_detector = None
        if not self.fsm.transition(self.calib_return):
            self.fsm.force(self.calib_return)
        if self.calib_return == S.MENU:
            self.main_menu.reset()

    # ------------------------------------------------------------------
    # Events / input
    # ------------------------------------------------------------------
    def handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            self.held.add(e.key)
            self.on_key(e.key)
        elif e.type == pygame.KEYUP:
            self.held.discard(e.key)
        elif e.type == pygame.WINDOWFOCUSLOST:
            self.held.clear()
        elif e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            pos = self.to_logical(e.pos)
            click = e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
            self.on_mouse(pos, click)
        elif e.type == pygame.MOUSEWHEEL and self.state == S.SETTINGS:
            self.settings_screen.mouse(self.to_logical(pygame.mouse.get_pos()), False, e.y)

    def on_mouse(self, pos, click: bool) -> None:
        st = self.state
        if st == S.MENU:
            r = self.main_menu.mouse(pos, click)
            if r is not None:
                self.main_menu_action(r)
        elif st == S.SETTINGS:
            self.settings_screen.mouse(pos, click)
        elif st == S.PAUSED:
            r = self.pause_menu.mouse(pos, click)
            if r is not None:
                self.pause_action(r)
        elif st in (S.VICTORY, S.DEFEAT):
            r = self.result_menu.mouse(pos, click)
            if r is not None:
                self.result_action(r)

    def main_menu_action(self, i: int) -> None:
        self.audio.play("ui")
        if i == 0:
            self.start_game(False)
        elif i == 1:
            self.start_game(True)
        elif i == 2:
            self.begin_calibration(S.MENU)
        elif i == 3:
            self.goto(S.SETTINGS)
        elif i == 4:
            self.show_help = True
        elif i == 5:
            self.running = False

    def pause_action(self, i: int) -> None:
        if i == 0:
            self.resume()
        elif i == 1:
            self.fsm.resume()
            self.restart_battle()
        else:
            self.goto(S.MENU)

    def result_action(self, i: int) -> None:
        b = self.battle
        if b is None:
            return
        if i == 0:
            if self.state == S.VICTORY:
                b.next_round()
                self.toasts.add(f"ROUND {b.round} - FIGHT!", ui.GOLD, 2.2)
            else:
                b.restart()
                self.toasts.add("REMATCH!", ui.GOLD)
            self.fx.reset()
            self.enemy.reset()
            self.detector.reset()
            self.input_mapper.reset()
            self.hand_engine.reset()
            self.fighter.reset()
        elif i == 1:
            if not self.fsm.transition(b.idle_state):
                self.fsm.force(b.idle_state)
            self.restart_battle()
        else:
            self.goto(S.MENU)

    def resume(self) -> None:
        if self.fsm.resume():
            if self.fsm.state == S.CHARGING:
                self.audio.start_loop("charge")
            elif getattr(self.battle, "ki_charging", False):
                self.audio.start_loop("kicharge")

    def on_key(self, key: int) -> None:
        if self.show_help:
            self.show_help = False
            return
        if key == pygame.K_f:
            self.toggle_fullscreen()
            return
        if key == pygame.K_m:
            self.audio.toggle_mute()
            self.toasts.add("MUTED" if self.settings.muted else "SOUND ON")
            return
        if key == pygame.K_F3:
            self.show_diag = not self.show_diag
            return
        if key == pygame.K_F4:
            self.pip_mode = not self.pip_mode
            self.toasts.add("FACE BOX VIEW" if self.pip_mode else "FULL WEBCAM VIEW")
            return
        st = self.state
        if st == S.MENU:
            if key == pygame.K_h:
                self.show_help = True
                return
            r = self.main_menu.key(key)
            if r is not None:
                self.main_menu_action(r)
            elif key == pygame.K_ESCAPE:
                self.running = False
        elif st == S.SETTINGS:
            if key == pygame.K_ESCAPE:
                self.goto(S.MENU)
            else:
                self.settings_screen.key(key)
        elif st == S.CALIBRATION:
            if key == pygame.K_ESCAPE:
                self.end_calibration(False)
            elif key == pygame.K_r and self.calibrator is not None:
                self.calibrator.restart()
                self.calib_detector = None
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.end_calibration(True)
        elif st == S.PAUSED:
            if key == pygame.K_ESCAPE:
                self.resume()
            elif key == pygame.K_r:
                self.fsm.resume()
                self.restart_battle()
            else:
                r = self.pause_menu.key(key)
                if r is not None:
                    self.pause_action(r)
        elif st in (S.VICTORY, S.DEFEAT):
            if key == pygame.K_ESCAPE:
                self.goto(S.MENU)
            elif key == pygame.K_r:
                self.result_action(1)
            else:
                r = self.result_menu.key(key)
                if r is not None:
                    self.result_action(r)
        elif st in COMBAT_STATES:
            if key == pygame.K_ESCAPE:
                if self.fsm.pause():
                    self.audio.stop_all_effects()
                    self.pause_menu.reset()
            elif key == pygame.K_r:
                self.restart_battle()
            elif key == pygame.K_h:
                self.show_help = True
            elif not self.face_only and key == pygame.K_1:
                self.kb.transform = True
            elif not self.face_only and key == pygame.K_3:
                self.kb.scan = True
            elif not self.face_only and key == pygame.K_4:
                self.kb.rage = True
            elif not self.face_only and key == pygame.K_6:
                self.kb.ultimate = True

    # ------------------------------------------------------------------
    # Per-frame input gathering
    # ------------------------------------------------------------------
    def poll_tracking(self, dt: float) -> None:
        pipe = self.pipeline
        res = pipe.latest() if pipe is not None else None
        self.track = res
        meas = res.measurements if res is not None else None
        use_face = pipe is not None
        det = self.detector
        if self.state == S.CALIBRATION and self.calibrator is not None:
            self.calibrator.update(meas, dt)
            if self.calibrator.finished and self.calib_detector is None:
                self.calib_detector = GestureDetector(self.calibrator.build(), self.settings.sensitivity,
                                                      self.settings.hold_ms / 1000, self.settings.invert_tilt)
            if self.calib_detector is not None:
                self.gframe = self.calib_detector.update(meas, dt)
            else:
                self.gframe = None
        elif use_face:
            self.gframe = det.update(meas, dt)
        else:
            self.gframe = None
        # anchor from landmarks
        target = None
        if res is not None and res.landmarks is not None:
            if self.mapper_fm is None or self.mapper_fm.fw != res.size[0] or self.mapper_fm.fh != res.size[1] \
                    or self.mapper_fm.mirror != res.mirror:
                self.mapper_fm = FrameMapper(res.size[0], res.size[1], mirror=res.mirror)
            target = compute_anchor(res.landmarks, self.mapper_fm)
        if target is not None:
            self.anchor = self.anchor_smoother.update(target, dt)
        elif res is not None and res.frame is not None:
            v = self.anchor_smoother.value
            v.present = False
            self.anchor = v
        else:
            self.anchor = PlayerAnchor.default()
            self.anchor.present = False
            self.anchor_smoother.value = self.anchor
        self.poll_hands()

    # -- hands -----------------------------------------------------------
    @property
    def hands_active(self) -> bool:
        """True while the hand tracker is running; then hands own charge / shield / ultimate."""
        return getattr(self.pipeline, "hand_status", "disabled") == "ok"

    def poll_hands(self) -> None:
        res = self.track
        obs: list[HandObs] = []
        if res is not None and res.hands is not None:
            if res.seq != self._hand_seq:           # only treat genuinely new camera frames as fresh
                self._hand_seq = res.seq
                self._hand_seen = self.t
            if self.t - self._hand_seen < 0.5:      # a frozen camera must not keep stale hands alive
                obs = res.hands
            if res.size[0] > 0:
                self.hand_engine.aspect = res.size[0] / max(1, res.size[1])
        self.hand_obs = obs

    def hand_to_screen(self, pt) -> tuple[float, float] | None:
        if pt is None or self.mapper_fm is None:
            return None
        return self.mapper_fm.to_screen(float(pt[0]), float(pt[1]))

    def cancel_charge(self) -> None:
        """Drop a charge without firing (tracking loss / switching to the ultimate pose)."""
        b = self.battle
        if b is not None and b.fsm.state == S.CHARGING:
            b.charge = 0.0
            b.fsm.transition(b.idle_state)
            b.emit("charge_cancel")

    def gather_commands(self, dt: float) -> Commands:
        h = self.held
        kb = self.kb
        self.kb = Commands()
        fo = self.face_only
        kb.charge_held = (not fo) and (pygame.K_SPACE in h or pygame.K_2 in h)
        kb.shield_held = (not fo) and (pygame.K_5 in h)
        up = (not fo) and (pygame.K_UP in h or pygame.K_w in h)
        down = (not fo) and (pygame.K_DOWN in h or pygame.K_s in h)
        if up != down:
            self.key_aim += (1.6 if up else -1.6) * dt
        else:
            self.key_aim -= math.copysign(min(abs(self.key_aim), 1.2 * dt), self.key_aim)
        self.key_aim = max(-1.0, min(1.0, self.key_aim))
        kb.aim = self.key_aim
        face = Commands()
        if self.settings.face_controls and self.gframe is not None and self.state in COMBAT_STATES:
            face = self.input_mapper.map(self.gframe, dt)
        hands_on = self.hands_active
        # KI CHARGE (power-up): hold C, or open your mouth while the hands are running the Ki Blast
        ki_held = ((not fo) and pygame.K_c in h) or (hands_on and bool(face.charge_held))
        if self.battle is not None:
            self.battle.ki_request = ki_held
        if hands_on:
            # hands own Ki Blast / shield / ultimate; the face keeps transform, rage, scan and aim
            face.charge_held = False
            face.shield_held = False
            face.ultimate = False
        cmds = kb.merged(face)
        if hands_on:
            hf = self.hand_engine.update(self.hand_obs, dt)
            self.hframe = hf
            self.hand_screen = self.hand_to_screen(hf.aim_point)
            if hf.cancel_charge:
                self.cancel_charge()
            cmds.charge_held = bool(cmds.charge_held or hf.charge_held)
            cmds.shield_held = bool(cmds.shield_held or hf.shield_held)
            cmds.ultimate = bool(cmds.ultimate or hf.ultimate)
        else:
            self.hframe = None
            self.hand_screen = None
        return cmds

    # ------------------------------------------------------------------
    # Battle events
    # ------------------------------------------------------------------
    def make_stage_anchor(self) -> PlayerAnchor:
        """Fixed fighting position on the left; the body leans a little with your real head position."""
        lean = 0.0
        if self.anchor.present:
            lean = max(-35.0, min(35.0, (self.anchor.head[0] - W / 2) * 0.12))
        return PlayerAnchor.from_params((STAGE_HEAD[0] + lean, STAGE_HEAD[1]), STAGE_FACE[0], STAGE_FACE[1], 0.0, True)

    def view_anchor(self) -> PlayerAnchor:
        """The anchor everything is drawn from: the stage fighter, or the real face in full-webcam view."""
        return self.stage if self.pip_mode else self.anchor

    def ctx(self) -> FxContext:
        if self.pip_mode:   # beams / orbs come from the fighter's hand
            anchor = dataclasses.replace(self.stage, hand=self.fighter.hand)
        else:               # full webcam view: from your real hand when it is visible
            anchor = self.anchor
            if self.hand_screen is not None and self.hand_engine.aim_age < 1.0:
                anchor = dataclasses.replace(anchor, hand=self.hand_screen)
        return FxContext(anchor, C.ENEMY_POS, self.battle)

    def on_event(self, ev: Event) -> None:
        self.event_counts[ev.kind] = self.event_counts.get(ev.kind, 0) + 1
        b = self.battle
        self.fx.handle_event(ev, self.ctx())
        self.audio.handle_event(ev)
        self.fighter.on_event(ev)
        k, d = ev.kind, ev.data
        used = {"transform_complete": Ability.TRANSFORM, "beam_fire": Ability.BLASTER, "scan_on": Ability.SCANNER,
                "rage_active": Ability.RAGE, "shield_on": Ability.SHIELD, "ultimate_fire": Ability.ULTIMATE}
        if k in used:
            self.used_abilities.add(used[k])
        if k == "transform_complete":
            self.banner.show("SUPER AWAKENING", f"POWER x{d['multiplier']:.1f}   LEVEL {d['power']:,}", ui.GOLD, 3.0)
        elif k == "rage_active":
            self.banner.show("RAGE BURST", f"DAMAGE x{d['multiplier']:.1f}", ui.RED, 2.2)
        elif k == "ultimate_start":
            self.banner.show("ULTIMATE FORM", "FINAL BLAST CHARGING...", (255, 240, 170), 2.4)
        elif k == "ki_charge_start":
            self.toasts.add("POWERING UP...", ui.BLUE, 1.0)
        elif k == "ultimate_ready":
            self.toasts.add("ULTIMATE READY!", ui.GOLD, 2.0)
        elif k == "denied" and d["ability"] == Ability.ULTIMATE and d["reason"] == "energy":
            self.toasts.add(f"ULTIMATE NEEDS {C.ULTIMATE_COST:.0f} KI - HOLD C / OPEN MOUTH TO CHARGE", ui.RED, 2.5)
        elif k == "denied":
            label = d["ability"].name.title()
            msg = ui.DENY_TEXT.get(d["reason"], d["reason"].upper())
            if d["reason"] == "cooldown" and d.get("remaining", 0) > 0:
                msg += f" ({d['remaining']:.0f}s)"
            self.toasts.add(f"{label}: {msg}", ui.RED)
        elif k == "enemy_stagger":
            self.toasts.add("ENEMY STAGGERED!", ui.GOLD)
        elif k == "player_dodged":
            self.toasts.add("INVULNERABLE", ui.BLUE)
        elif k == "beam_miss":
            self.toasts.add("MISS - ADJUST AIM", ui.GREY)
        elif k == "transform_end":
            self.toasts.add("AWAKENING FADED", ui.GREY)
        elif k == "shield_break":
            self.toasts.add("SHIELD BROKEN", ui.RED)
        elif k in ("victory", "defeat"):
            self.result_menu.items = (["NEXT ROUND", "RESTART", "MAIN MENU"] if k == "victory"
                                      else ["REMATCH", "RESTART", "MAIN MENU"])
            self.result_menu.reset()
            self.result_menu.top = 360

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        self.t += dt
        self.audio.update(dt)
        self.run_script()
        if self.pending_cam_restart is not None:
            self.pending_cam_restart -= dt
            if self.pending_cam_restart <= 0:
                self.pending_cam_restart = None
                self.restart_camera()
        self.poll_tracking(dt)
        self.banner.update(dt)
        self.toasts.update(dt)
        st = self.state
        b = self.battle
        if st not in COMBAT_STATES:
            self.hand_engine.reset()       # no gesture state may leak across pause / victory / menus
            self.hframe = None
            self.hand_screen = None
        if b is not None and st != S.PAUSED:
            if st in COMBAT_STATES:
                cmds = self.gather_commands(dt)
                b.apply_commands(cmds, self.settings.brow_action)
                b.update(dt)
            self.stage = self.make_stage_anchor()
            sc = self.gframe.scores if self.gframe else {}
            self.fighter.update(dt, b, self.stage, C.ENEMY_POS,
                                dict(brow_up=float(sc.get(Gesture.BROW_RAISE, 0)), brow_down=float(sc.get(Gesture.BROW_LOWER, 0)),
                                     mouth=float(sc.get(Gesture.MOUTH_OPEN, 0)), purse=float(sc.get(Gesture.PURSE, 0))),
                                self.fx.hair_alpha)          # pose is solved BEFORE events so beams start at the hand
            for ev in b.drain_events():
                self.on_event(ev)
            self.fx.update(dt, self.ctx())
            self.enemy.update(dt, b)
            self.scanner.update(dt, b)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def draw_camera(self, target: pygame.Surface, size=None, dest=(0, 0)) -> bool:
        """Draw the webcam frame. Defaults to full screen; pass size/dest for a small box."""
        res = self.track
        if res is None or res.frame is None:
            return False
        if res.seq != self._cam_seq or self._cam_src is None:
            fh, fw = res.frame.shape[:2]
            self._cam_src = pygame.image.frombuffer(res.frame, (fw, fh), "RGB")
            self._cam_seq = res.seq
            if self.mapper_fm is None or self.mapper_fm.fw != fw or self.mapper_fm.fh != fh:
                self.mapper_fm = FrameMapper(fw, fh, mirror=res.mirror)
        x, y, w, h = self.mapper_fm.crop_rect
        try:
            cropped = self._cam_src.subsurface((x, y, w, h))
            scaled = pygame.transform.scale(cropped, size or (W, H))
            target.blit(scaled, dest)
        except (ValueError, pygame.error) as exc:
            log.warning("Camera frame could not be drawn: %s", exc)
            self._cam_src = None
            return False
        return True

    def draw_world(self) -> None:
        w = self.world
        b = self.battle
        has_cam = False if self.pip_mode else self.draw_camera(w)
        if self.pip_mode:
            light = None
            if self.fx.aura_intensity > 0.05:
                light = (self.fx.aura_style.mid, self.fx.aura_intensity, self.stage.aura_center)
            self.arena.draw(w, self.t, light)
        elif not has_cam:
            self.bg.draw(w, self.t)
        else:
            w.fill((200, 200, 215), special_flags=pygame.BLEND_RGB_MULT)  # slight darkening for contrast
        if not self.pip_mode and (self.settings.show_landmarks or self.show_diag):
            res = self.track
            if res is not None and res.landmarks is not None and self.mapper_fm is not None:
                ui.draw_landmarks(w, res.landmarks, self.mapper_fm)
        self.fx.draw_dim(w)
        self.enemy.draw(w, C.ENEMY_POS, b)
        if self.pip_mode:
            self.fighter.draw(w)                     # side-view fighter facing the enemy
        elif not has_cam:
            f = self.gframe
            sc = f.scores if f else {}
            self.avatar.draw(w, self.anchor, self.t, sc.get(Gesture.BROW_RAISE, 0), sc.get(Gesture.BROW_LOWER, 0),
                             sc.get(Gesture.MOUTH_OPEN, 0), sc.get(Gesture.PURSE, 0), plain_hair=self.fx.hair_alpha < 0.95)
        if self.fx.hair_alpha > 0.01 and not self.pip_mode:   # the fighter draws its own hair
            tint = None
            if b is not None:
                tint = "red" if b.rage_time > 0 else ("white" if b.fsm.state == S.ULTIMATE else None)
            self.hair.draw(w, self.anchor, self.fx.hair_alpha, tint)
        self.fx.draw(w, self.ctx())

    def tracking_info(self):
        return ui.tracking_label(self.pipeline, self.gframe, self.settings.face_controls)

    def draw(self) -> None:
        s = self.screen
        st = self.state
        dt = 1 / 60
        if st == S.MENU:
            self.bg.draw(s, self.t)
            ui.draw_title(s, self.t)
            self.main_menu.draw(s, dt, self.t)
            txt, col = self.tracking_info()
            ui.text(s, txt, (W // 2, H - 36), 22, col, "center")
            ui.text(s, "UP/DOWN + ENTER, or click.  H = controls", (W // 2, H - 64), 20, ui.GREY, "center")
        elif st == S.SETTINGS:
            self.bg.draw(s, self.t)
            self.settings_screen.draw(s, self.t)
        elif st == S.CALIBRATION:
            if not self.draw_camera(s):
                self.bg.draw(s, self.t)
            elif self.track is not None and self.track.landmarks is not None and self.mapper_fm is not None:
                ui.draw_landmarks(s, self.track.landmarks, self.mapper_fm)
            if self.calibrator is not None:
                ok = self.pipeline is not None and (self.pipeline.status == "running")
                ui.draw_calibration(s, self.calibrator, self.gframe, self.tracking_info(), self.t, ok)
        else:
            self.draw_world()
            s.fill((0, 0, 0))
            s.blit(self.world, self.fx.shake_offset())
            b = self.battle
            if b is not None:
                training = b.training
                ui.draw_scanner(s, b, self.view_anchor(), C.ENEMY_POS, self.t, self.scanner)
                ui.draw_hud(s, b, self.t, self.gframe, self.tracking_info(), training, self.fps, True, self.enemy.NAME)
                if training:
                    self.draw_training_panel(s)
                self.banner.draw(s)
                self.toasts.draw(s)
                if self.pip_mode and st != S.PAUSED:
                    self.scouter.draw(s, self)
                if st == S.PAUSED:
                    ui.dim_overlay(s, 170)
                    ui.text(s, "PAUSED", (W // 2, 200), 110, ui.GOLD, "center")
                    self.pause_menu.draw(s, dt, self.t)
                elif st in (S.VICTORY, S.DEFEAT):
                    ui.draw_result(s, st == S.VICTORY, b, self.t, self.result_menu, dt)
        if self.show_diag:
            meas = self.track.measurements if self.track is not None else None
            lines = [f"cam: {getattr(self.pipeline, 'status', 'off')}  tracker: {getattr(self.pipeline, 'tracker_status', '-')}"]
            if self.battle is not None:
                lines.append(f"battle: hp {self.battle.player_hp:.0f} en {self.battle.energy:.0f} aim {self.battle.aim:+.2f}")
                lines.append(f"particles: {self.fx.particles.count}/{self.fx.particles.capacity}")
            ui.draw_diagnostics(s, meas, self.gframe, st.name, self.fps, self.calibration, lines)
        if self.show_help:
            scouter_ui.draw_controls_help(s, self.t)

    def draw_training_panel(self, s) -> None:
        ui.panel(s, (W - 436, 104, 420, 74), 150)
        ui.text(s, "TRAINING - USE EACH ABILITY", (W - 424, 110), 20, ui.GREY)
        for i, a in enumerate(Ability):
            used = a in self.used_abilities
            ui.text(s, f"{int(a)} {a.name.title()}", (W - 424 + (i % 3) * 136, 136 + (i // 3) * 22), 20,
                    ui.GREEN if used else ui.WHITE)

    # ------------------------------------------------------------------
    # Self-test (scripted automatic playthrough)
    # ------------------------------------------------------------------
    def _setup_selftest(self) -> None:
        if self.args.shots:
            self.shots_dir = Path(self.args.shots)
            self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.settings.first_run_done = True
        self.start_game(False)
        self.battle.rng.seed(3)
        if self.args.demo_face:  # gestures drive the game; no keyboard script
            return
        self.script = [
            (1.0, "key:1"), (4.2, "shot:transform"), (5.0, "down:space"), (5.9, "shot:charge"), (6.5, "up:space"),
            (6.75, "shot:beam"), (7.6, "key:3"), (8.2, "shot:scan"), (9.0, "key:4"), (10.1, "shot:rage"),
            (11.0, "down:5"), (13.5, "shot:shield"), (14.5, "up:5"), (15.0, "down:space"), (16.6, "up:space"),
            (17.5, "hook:fullenergy"), (17.6, "key:6"), (19.0, "shot:ultimate"), (21.0, "shot:ultimate_beam"),
            (23.5, "key:esc"), (24.0, "shot:paused"), (24.5, "key:esc"), (25.0, "hook:kill_enemy"), (26.0, "shot:victory"),
        ]

    def run_script(self) -> None:
        while self.script_i < len(self.script) and self.t >= self.script[self.script_i][0]:
            _, act = self.script[self.script_i]
            self.script_i += 1
            kind, _, arg = act.partition(":")
            keymap = {"1": pygame.K_1, "3": pygame.K_3, "4": pygame.K_4, "5": pygame.K_5, "6": pygame.K_6,
                      "space": pygame.K_SPACE, "esc": pygame.K_ESCAPE}
            if kind == "key":
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=keymap[arg]))
                self.handle_event(pygame.event.Event(pygame.KEYUP, key=keymap[arg]))
            elif kind == "down":
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=keymap[arg]))
            elif kind == "up":
                self.handle_event(pygame.event.Event(pygame.KEYUP, key=keymap[arg]))
            elif kind == "hook" and self.battle is not None:  # test-only shortcuts
                if arg == "fullenergy":
                    self.battle.energy = C.PLAYER_MAX_ENERGY
                    self.battle.player_hp = C.PLAYER_MAX_HP
                elif arg == "kill_enemy":
                    self.battle.enemy_hp = 1.0
                    self.battle.damage_enemy(5, "selftest")
            elif kind == "shot":
                self.save_shot(arg)

    def save_shot(self, name: str) -> None:
        if self.shots_dir is not None:
            self.draw()
            pygame.image.save(self.screen, str(self.shots_dir / f"{name}.png"))

    # ------------------------------------------------------------------
    def run(self) -> int:
        clock = pygame.time.Clock()
        fps_samples: list[float] = []
        while self.running:
            raw = clock.tick(C.TARGET_FPS) / 1000.0
            dt = min(raw, C.MAX_DT)
            self.fps = clock.get_fps()
            for e in pygame.event.get():
                self.handle_event(e)
            try:
                self.update(dt)
                self.draw()
            except Exception:
                log.exception("Frame error (continuing)")
            self.present()
            if self.args.selftest:
                if self.t > 1.0 and self.fps > 0:
                    fps_samples.append(self.fps)
                if self.t >= self.args.selftest:
                    self.running = False
        if self.args.selftest:
            self.report_selftest(fps_samples)
        self.shutdown()
        return 0

    def report_selftest(self, fps: list[float]) -> None:
        b = self.battle
        print("SELFTEST RESULT")
        print("  final state :", self.state.name)
        print("  events      :", dict(sorted(self.event_counts.items())))
        if b is not None:
            print(f"  round={b.round} score={b.score} hp={b.player_hp:.0f} energy={b.energy:.0f}")
        if fps:
            print(f"  fps (headless, dummy video, this machine): avg={sum(fps) / len(fps):.1f} min={min(fps):.1f}")
        print("  abilities used:", sorted(a.name for a in self.used_abilities))

    def shutdown(self) -> None:
        try:
            self.settings.save() if not self.args.selftest else None
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
            self.audio.shutdown()
            pygame.quit()


def main() -> int:
    logging.basicConfig(level=logging.DEBUG if ARGS.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        app = App(ARGS)
    except Exception:
        log.exception("Fatal error while starting. Run: python scripts/check_environment.py")
        return 1
    return app.run()


if __name__ == "__main__":
    sys.exit(main())