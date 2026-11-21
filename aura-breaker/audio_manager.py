"""Audio manager.

Sounds are the original procedurally synthesized WAV files from ``sound_design.py``.
A missing sound file or a missing audio device is never fatal: the game simply runs silently.

Improvements over the first version:
  * every event of the combat loop has its own sound (charge, Ki Charge, beams, ultimate charge and beam,
    enemy telegraph / attack, hit reactions, shield on / hit / break, ready chime, power-down ...),
  * music switches between a menu theme and a battle theme, and DUCKS under big moments
    (transformation, ultimate, victory) so they hit harder,
  * looping sounds fade in / out, can never be started twice, and a watchdog clears loops that were
    stolen by the mixer, so nothing gets stuck after pause / restart / state changes,
  * one-shots are rate limited and capped per sound, so a gesture that flickers cannot machine-gun the speakers.
"""
from __future__ import annotations

import logging
import time

import pygame

import config as C
from battle_system import Event

log = logging.getLogger("aura.audio")

# name -> (volume category, filename).  Categories must exist in config.VOLUME_CHANNELS.
SOUND_MAP: dict[str, tuple[str, str]] = {
    "music": ("music", "music.wav"),
    "music_menu": ("music", "music_menu.wav"),
    "charge": ("charge", "charge.wav"),
    "charge_start": ("charge", "charge_start.wav"),
    "kicharge": ("charge", "kicharge.wav"),
    "fizzle": ("charge", "fizzle.wav"),
    "beam": ("beam", "beam.wav"),
    "ultimate_beam": ("beam", "ultimate_beam.wav"),
    "impact": ("impact", "impact.wav"),
    "hurt": ("impact", "hurt.wav"),
    "warn": ("impact", "warn.wav"),
    "eattack": ("impact", "eattack.wav"),
    "transform": ("transform", "transform.wav"),
    "powerdown": ("transform", "powerdown.wav"),
    "rage": ("transform", "rage.wav"),
    "ultimate": ("transform", "ultimate.wav"),
    "ready": ("transform", "ready.wav"),
    "shield": ("shield", "shield.wav"),
    "shield_hit": ("shield", "shield_hit.wav"),
    "shield_break": ("shield", "shield_break.wav"),
    "scan": ("shield", "scan.wav"),
    "ui": ("shield", "ui.wav"),
    "deny": ("shield", "deny.wav"),
    "victory": ("result", "victory.wav"),
    "defeat": ("result", "defeat.wav"),
}

MAX_VOICES = {"impact": 3, "hurt": 2, "shield_hit": 3, "beam": 3, "ui": 2, "deny": 1, "warn": 1, "eattack": 2}
MIN_GAP = {"deny": 0.35, "impact": 0.04, "shield_hit": 0.05, "hurt": 0.10, "ui": 0.05, "ready": 0.5, "charge_start": 0.25}
LOOPS = ("charge", "kicharge")


class AudioManager:
    def __init__(self, settings: C.Settings) -> None:
        self.settings = settings
        self.enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self._music_ch = None
        self._music_name = ""
        self._loop_ch: dict[str, pygame.mixer.Channel] = {}
        self._last: dict[str, float] = {}
        self._duck = 1.0
        self._duck_target = 1.0
        self._duck_timer = 0.0
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            pygame.mixer.set_num_channels(24)
            pygame.mixer.set_reserved(1)
            self._music_ch = pygame.mixer.Channel(0)
            self.enabled = True
        except pygame.error as exc:
            log.warning("Audio disabled (no mixer): %s", exc)
            return
        for name, (_, fname) in SOUND_MAP.items():
            path = C.SOUND_DIR / fname
            if not path.exists():
                log.warning("Sound file missing (continuing silently): %s", path)
                continue
            try:
                self.sounds[name] = pygame.mixer.Sound(str(path))
            except pygame.error as exc:
                log.warning("Could not load %s: %s", path, exc)

    # ------------------------------------------------------------------
    def _vol(self, name: str) -> float:
        if self.settings.muted:
            return 0.0
        cat = SOUND_MAP[name][0]
        v = self.settings.master_volume * self.settings.volumes.get(cat, 0.8)
        return v * self._duck if cat == "music" else v

    def play(self, name: str) -> None:
        snd = self.sounds.get(name)
        if not self.enabled or snd is None:
            return
        now = time.monotonic()
        if now - self._last.get(name, -9.0) < MIN_GAP.get(name, 0.0):
            return
        if snd.get_num_channels() >= MAX_VOICES.get(name, 4):
            return
        self._last[name] = now
        snd.set_volume(self._vol(name))
        try:
            snd.play()
        except pygame.error as exc:
            log.debug("play %s: %s", name, exc)

    # -- loops -------------------------------------------------------------
    def _loop_alive(self, name: str) -> bool:
        ch = self._loop_ch.get(name)
        if ch is None:
            return False
        try:
            return bool(ch.get_busy() and ch.get_sound() is self.sounds.get(name))
        except pygame.error:
            return False

    def start_loop(self, name: str, fade_ms: int = 160) -> None:
        snd = self.sounds.get(name)
        if not self.enabled or snd is None:
            return
        if self._loop_alive(name):
            return
        self._loop_ch.pop(name, None)
        snd.set_volume(self._vol(name))
        try:
            ch = snd.play(loops=-1, fade_ms=fade_ms)
        except pygame.error:
            return
        if ch is not None:
            self._loop_ch[name] = ch

    def stop_loop(self, name: str, fade_ms: int = 220) -> None:
        ch = self._loop_ch.pop(name, None)
        if ch is not None:
            try:
                ch.fadeout(fade_ms)
            except pygame.error:
                pass

    def stop_all_effects(self) -> None:
        for name in list(self._loop_ch):
            self.stop_loop(name, 80)

    # -- music -------------------------------------------------------------
    def play_music(self, name: str = "music") -> None:
        snd = self.sounds.get(name) or self.sounds.get("music")
        if not self.enabled or snd is None or self._music_ch is None:
            return
        if self._music_name == name and self._music_ch.get_busy():
            self.apply_settings()
            return
        self._music_ch.stop()
        self._music_ch.play(snd, loops=-1, fade_ms=900)
        self._music_name = name
        self.apply_settings()

    def duck(self, amount: float = 0.35, seconds: float = 3.0) -> None:
        """Lower the music briefly so a big sound effect cuts through."""
        self._duck_target = max(0.05, min(1.0, amount))
        self._duck_timer = max(self._duck_timer, seconds)

    # ------------------------------------------------------------------
    def apply_settings(self) -> None:
        if not self.enabled:
            return
        if self._music_ch is not None and self._music_name:
            self._music_ch.set_volume(self._vol(self._music_name))
        for name, ch in list(self._loop_ch.items()):
            try:
                ch.set_volume(self._vol(name))
            except pygame.error:
                pass

    def update(self, dt: float) -> None:
        """Call once per frame: eases the music duck and clears dead loop handles."""
        if not self.enabled:
            return
        if self._duck_timer > 0:
            self._duck_timer -= dt
            if self._duck_timer <= 0:
                self._duck_target = 1.0
        k = 1 - pow(0.001, dt)            # fast attack, smooth release
        self._duck += (self._duck_target - self._duck) * (k if self._duck_target < self._duck else k * 0.35)
        if self._music_ch is not None and self._music_name:
            self._music_ch.set_volume(self._vol(self._music_name))
        for name in list(self._loop_ch):
            if not self._loop_alive(name):
                self._loop_ch.pop(name, None)

    def toggle_mute(self) -> bool:
        self.settings.muted = not self.settings.muted
        self.apply_settings()
        return self.settings.muted

    def shutdown(self) -> None:
        if self.enabled:
            try:
                pygame.mixer.stop()
            except pygame.error:
                pass

    # ------------------------------------------------------------------
    def handle_event(self, ev: Event) -> None:
        k, d = ev.kind, ev.data
        if k == "transform_start":
            self.stop_loop("kicharge")
            self.play("transform")
            self.duck(0.3, 3.9)
        elif k == "transform_end":
            self.play("powerdown")
        elif k == "rage_start":
            self.play("rage")
            self.duck(0.4, 1.6)
        elif k == "charge_start":
            self.play("charge_start")
            self.start_loop("charge")
        elif k == "charge_cancel":
            self.stop_loop("charge")
            self.play("fizzle")
        elif k == "ki_charge_start":
            self.play("charge_start")
            self.start_loop("kicharge")
        elif k == "ki_charge_stop":
            self.stop_loop("kicharge")
        elif k == "beam_fire":
            self.stop_loop("charge")
            self.play("beam")
        elif k == "ultimate_start":
            self.stop_loop("charge")
            self.stop_loop("kicharge")
            self.play("ultimate")
            self.duck(0.22, 5.2)
        elif k == "ultimate_fire":
            self.play("ultimate_beam")
        elif k == "ultimate_ready":
            self.play("ready")
        elif k == "enemy_telegraph":
            self.play("warn")
        elif k == "enemy_attack":
            self.play("eattack")
        elif k == "enemy_hit":
            self.play("impact")
        elif k == "player_hit":
            if d.get("damage", 0) > 0:
                self.play("hurt")
        elif k == "shield_on":
            self.play("shield")
        elif k == "shield_hit":
            self.play("shield_hit")
        elif k == "shield_break":
            self.play("shield_break")
        elif k == "scan_on":
            self.play("scan")
        elif k == "denied":
            self.play("deny")
        elif k == "victory":
            self.stop_all_effects()
            self.play("victory")
            self.duck(0.25, 4.5)
        elif k == "defeat":
            self.stop_all_effects()
            self.play("defeat")
            self.duck(0.25, 4.0)