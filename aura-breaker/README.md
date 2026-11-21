# AURA BREAKER: SAIYAN AWAKENING
**YOUR FACE. YOUR POWER. YOUR BATTLE.**

A face-controlled anime-style battle game for macOS (also runs on Linux/Windows). Your webcam and
MediaPipe Face Landmarker turn facial gestures into transformations, energy blasts, shields and an
ultimate attack against a virtual enemy. All artwork, hair, sound and music are **original and
procedurally generated** - no Dragon Ball characters, logos, music or sound effects are used.

---------------------------------------------------------------------------------------------------
## 1. Install (macOS, VS Code terminal)

```bash
mkdir -p ~/Documents/xample && cd ~/Documents/xample
# unzip aura-breaker.zip here, so you get ~/Documents/xample/aura-breaker
cd ~/Documents/xample/aura-breaker

python3 --version            # use 3.10 - 3.12 (3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/setup_assets.py        # generates hair + sounds, downloads the MediaPipe face model
python scripts/check_environment.py --camera
```

If `python3 --version` is 3.13+ and `pip install` fails for mediapipe, install a compatible Python
(`brew install python@3.12`) and create the venv with `python3.12 -m venv .venv`.

If the model download fails (firewall), save it manually:
```bash
curl -L -o assets/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**Camera permission (macOS):** the first launch asks for camera access. If you denied it: System
Settings > Privacy & Security > Camera > enable your terminal / Visual Studio Code, then fully quit
and reopen it.

## 2. Run

```bash
cd ~/Documents/xample/aura-breaker
source .venv/bin/activate
python main.py
```

Useful options:

| Command | Purpose |
|---|---|
| `python main.py --no-camera` | keyboard-only play (no webcam needed) |
| `python main.py --demo-face` | scripted simulated face, to see the gesture pipeline without a webcam |
| `python main.py --diagnostic` | diagnostics overlay (also F3 in game) |
| `python main.py --quality low` | low-effects mode for slower Macs |
| `python main.py --camera 1` | use another camera (e.g. an external webcam) |
| `python main.py --selftest 27 --shots /tmp/shots` | headless scripted playthrough + screenshots |
| `python -m pytest tests -q` | automated tests (no webcam needed) |

On first launch with a camera, the **calibration** screen opens automatically (neutral face, open
mouth, brows up, brows down, lips pursed, then a test step). You can skip it with ESC (defaults apply)
or redo it from the menu / Settings.

## 3. Controls

| Face gesture (calibrated) | Action |
|---|---|
| Brows pulled down | Super Transformation; once transformed (or on cooldown) it triggers Rage Burst. Priority is configurable: Settings > Brow gesture action (AUTO / TRANSFORM / RAGE) |
| Mouth open (hold) | Charge Energy Blaster; close mouth to fire (auto-fires at full charge) |
| Brows raised (hold ~0.25 s) | Power Scanner |
| Brows raised + mouth open (hold 0.6 s) | Ultimate Form (needs 90 energy) |
| Lips pursed (hold) | Energy Shield |
| Tilt head | Aim (left shoulder = up, right = down; invertible in Settings) |

| Key | Action |
|---|---|
| 1 / 3 / 4 / 6 | Transformation / Scanner / Rage Burst / Ultimate |
| hold 2 or SPACE | charge Blaster, release to fire |
| hold 5 | Shield |
| UP/DOWN or W/S | aim |
| ESC | pause / resume (menus: back) |
| ENTER | confirm |
| R | restart battle |
| F | fullscreen |
| M | mute |
| F3 | diagnostics |
| H | controls help |

Gestures are *geometric measurements* (mouth gap, brow-to-eye distance, lip width, head roll) - not
emotion recognition. They are normalized by face size, smoothed, hysteresis-gated, hold-time
debounced, edge-triggered with cooldowns, and require a neutral face before triggering.

## 4. Abilities and balance (all constants in `config.py`)

| # | Ability | Cost | Effect |
|---|---|---|---|
| 1 | Super Transformation | 30 energy, 14 s cooldown | 2.6 s cinematic (invulnerable), then x1.5 damage for 22 s, drains 3 energy/s (no regen) |
| 2 | Energy Blaster | 10-25 energy by charge | 8-30 damage by charge; hit only if aim is within tolerance of the enemy |
| 3 | Power Scanner | 5 energy | 5 s HUD with real game values (power level = f(energy, multipliers, upgrades)) |
| 4 | Rage Burst | 25 energy, 18 s cooldown | red aura, x1.6 damage for 8 s, stacks with transformation |
| 5 | Energy Shield | drains 14 energy/s | absorbs 70% of damage using a 60-point shield that breaks, locks for 3 s, then regenerates |
| 6 | Ultimate Form | 90 energy, 45 s cooldown | 3.2 s sequence, 70 base damage (always hits), invulnerable during it |

Battle: telegraphed enemy attacks (projectile you can shield or dodge via a locked animation), strong hits
stagger a telegraphing enemy, combo/score system, victory -> next round (stronger enemy, +8% player power),
defeat -> rematch. Training mode has a non-attacking dummy, faster energy regen and an ability checklist.

## 5. Project layout

`main.py` loop/screens/input - `config.py` constants + settings - `game_state.py` state machine -
`battle_system.py` pure battle logic - `face_controls.py` landmarks -> gestures -> commands + calibration -
`face_tracking.py` MediaPipe Tasks API + camera thread - `camera.py` - `effects.py` aura/lightning/particles/beam/shield/hair -
`animation.py` enemy + avatar - `audio_manager.py` - `ui.py` - `scripts/` - `tests/`.

Replace the enemy by dropping a transparent PNG at `assets/images/enemy.png`. Replace the hair by overwriting
`assets/images/hair_gold.png` (anchor: forehead; designed so the hairline sits at 50% x, 74% y of the image).

## 6. Verification status (honest report)

Tested in a Linux sandbox (Python 3.12.3, mediapipe 1.1.1, opencv 5.0.0, numpy 2.5.4, pygame 2.6.1):
* `pytest`: **116 tests passed** (gestures, calibration, cooldowns, state machine, damage, energy, shield, ultimate
  requirements, victory/defeat, pause/restart, boundary values, effects clipping/pool caps).
* Fresh virtualenv install from `requirements.txt` + tests: passed.
* `main.py --selftest`: scripted keyboard playthrough exercising all six abilities, enemy attacks, shield, victory,
  pause - passed. Screenshots were inspected visually.
* `main.py --selftest --demo-face`: the *real* gesture detector and mapper drove transform, blaster, scanner and shield.
* No-camera / missing-model paths: game starts, shows a clear message, keyboard play works.
* Measured speed: ~52-58 FPS average on a 1-core sandbox CPU with a *dummy* video driver (no real window
  presentation cost). Not a MacBook number - check yours with F3 / `--diagnostic` and use `--quality low` if needed.

**NOT tested (no hardware/network in the sandbox):** a real webcam, macOS, actual MediaPipe inference (the model
download host was unreachable), real audio output, fullscreen. The face-landmark index math is verified with synthetic
landmarks only. Please run `python scripts/check_environment.py --camera` and the calibration screen on your Mac
and use F3 diagnostics to confirm the gesture values move as you make the faces.

## 7. Limitations

* The six "modes" are abilities selectable in Battle and Training (keys 1-6 or gestures), not six separate screens.
* No body segmentation: the aura is attached to an ellipse estimated from the tracked face (head + shoulders
  approximation). Hair overlay alignment degrades with large head turns, occlusion, hats or glasses glare.
* Brow-down vs. brow-up depends on lighting and face; recalibrate or adjust Face sensitivity if unreliable.
* Pursed lips is the least reliable gesture; use key 5 as a fallback.
* Camera selection is an index cycle (0-3), not a named device list.
* Only one face is tracked.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| "Could not open camera" | Grant camera permission to Terminal/VS Code, quit and reopen it; try `--camera 1`; close Zoom/FaceTime |
| "NO FACE MODEL" | `python scripts/setup_assets.py` (or manual curl above) |
| `pip install` fails on mediapipe | Use Python 3.10-3.12 |
| Gestures fire too easily / never | Settings > Face sensitivity, Gesture hold time, or recalibrate (better lighting, face the camera) |
| Low FPS | `--quality low`, close other camera apps, lower resolution is not needed (inference runs on a 640 px copy) |
| No sound | Check M / Settings volumes; the game runs silently if audio init fails |
| Reset everything | delete `settings.json` |
