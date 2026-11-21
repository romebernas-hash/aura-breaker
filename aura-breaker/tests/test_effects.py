import math
import random

import pygame
import pytest

import config as C
from battle_system import Battle, Event
from effects import (AURA_STYLES, Beam, Fx, FxContext, HairOverlay, LightningField, ParticleSystem, ScreenFx,
                     Shockwave, aim_geometry, draw_aura, lightning_path, make_hair_surface, P)
from face_tracking import PlayerAnchor
from game_state import GameState as S


@pytest.fixture(scope="module", autouse=True)
def pg():
    pygame.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.quit()


def ctx(battle=None, anchor=None):
    return FxContext(anchor or PlayerAnchor.default(), C.ENEMY_POS, battle)


def test_particle_pool_is_hard_capped():
    ps = ParticleSystem(50)
    assert ps.emit(500, (10, 10)) == 50
    assert ps.count == 50 and ps.emit(10, (0, 0)) == 0


def test_particles_expire_and_move():
    ps = ParticleSystem(20)
    ps.emit(5, (100, 100), speed=(100, 100), life=(0.2, 0.2))
    p0 = ps.pos[ps.alive].copy()
    ps.update(0.1)
    assert not (ps.pos[ps.alive] == p0).all()
    ps.update(0.5)
    assert ps.count == 0


def test_particle_attraction_pulls_toward_target():
    ps = ParticleSystem(10)
    ps.emit(1, (0, 0), speed=(0, 0), life=(2, 2), attract=5.0, target=(100, 0))
    for _ in range(30):
        ps.update(1 / 60)
    assert ps.pos[ps.alive][0][0] > 5


def test_particles_survive_nan_positions_in_render():
    ps = ParticleSystem(10)
    ps.emit(3, (50, 50))
    ps.pos[:] = float("nan")
    ps.update(0.01)
    ps.render(pygame.Surface((100, 100)))


def test_lightning_endpoints_and_point_count():
    rng = random.Random(1)
    path = lightning_path((0, 0), (100, 50), rng, depth=4)
    assert path[0] == (0.0, 0.0) and path[-1] == (100.0, 50.0) and len(path) == 2 ** 4 + 1


def test_lightning_regenerates_and_stays_near_aura():
    lf = LightningField(random.Random(3))
    center, radii = (300, 300), (200, 300)
    lf.update(1.0, center, radii, 1.0, 4)
    first = [b[0][0] for b in lf.bolts]
    assert 1 <= len(lf.bolts) <= 8
    for path, *_ in lf.bolts:
        for x, y in path:
            assert math.dist((x, y), center) < 900
    lf.update(1.0, center, radii, 1.0, 4)
    assert [b[0][0] for b in lf.bolts] != first               # animates between frames
    lf.update(1.0, center, radii, 0.0, 4)
    assert lf.bolts == []


def test_draw_functions_clip_and_survive_extreme_coordinates():
    s = pygame.Surface((320, 180))
    draw_aura(s, AURA_STYLES["gold"], (1e7, -1e7), (1e6, 1e6), 1.5, 3.0, 12)
    draw_aura(s, AURA_STYLES["red"], (float("nan"), 0), (50, 50), 1.0, 0.0, 8)
    Shockwave((-5000, 9e9), (255, 255, 255), 1e9).render(s)
    assert P(float("inf"), 1e12) == (0, 0)
    assert P(1e12, 1e12) == (6000, 6000)
    assert P(float("nan"), 1) == (0, 0)


def test_beam_reaches_impact_once_and_expires():
    b = Beam((0, 0), 0.0, 1000, 200, AURA_STYLES["blue"], 40, 0.5)
    hits = sum(b.update(1 / 60) for _ in range(60))
    assert hits == 1 and not b.alive and b.head == 200
    s = pygame.Surface((320, 180))
    Beam((0, 0), 0.3, 500, None, AURA_STYLES["blue"], 40, 0.5).render(s)


def test_beam_follows_aim_direction_not_fixed_point():
    a = PlayerAnchor.default()
    _, up, _ = aim_geometry(a, C.ENEMY_POS, 1.0)
    _, mid, _ = aim_geometry(a, C.ENEMY_POS, 0.0)
    _, down, _ = aim_geometry(a, C.ENEMY_POS, -1.0)
    assert up < mid < down
    assert mid - up == pytest.approx(math.radians(C.MAX_AIM_DEG))


def test_aim_geometry_zero_aim_points_at_enemy():
    a = PlayerAnchor.default()
    muzzle, angle, dist = aim_geometry(a, C.ENEMY_POS, 0.0)
    ox, oy = a.mouth[0] + a.face_w * 0.15, a.mouth[1]
    assert angle == pytest.approx(math.atan2(C.ENEMY_POS[1] - oy, C.ENEMY_POS[0] - ox))
    assert dist > 100


def test_screenfx_low_effects_scaling_and_decay():
    sc = ScreenFx()
    sc.scale = 0.5
    sc.trigger_flash(1.0)
    sc.add_shake(10)
    assert sc.flash == 0.5 and sc.shake == 5
    for _ in range(120):
        sc.update(1 / 60)
    assert sc.flash == 0 and sc.shake == 0
    assert sc.offset(random.Random()) == (0, 0)


@pytest.mark.parametrize("q", ["low", "standard", "high"])
def test_quality_limits_particles(q):
    fx = Fx(q, size=(320, 180), seed=1)
    assert fx.particles.capacity == C.QUALITY[q]["particles"]
    for _ in range(200):
        fx.burst((50, 50), (255, 0, 0), (255, 255, 0), 500)
    assert fx.particles.count <= fx.particles.capacity


def test_low_quality_reduces_bolts_and_flash():
    lo, hi = Fx("low", (320, 180)), Fx("high", (320, 180))
    assert C.QUALITY["low"]["bolts"] < C.QUALITY["high"]["bolts"]
    assert lo.screen.scale < hi.screen.scale


def test_fx_full_battle_flow_updates_and_renders_without_error():
    b = Battle(rng=random.Random(1))
    fx = Fx("standard", size=(320, 180), seed=2)
    world = pygame.Surface((320, 180))
    c = ctx(b)
    b.request_transform()
    for ev in b.drain_events():
        fx.handle_event(ev, c)
    for _ in range(int((C.TRANSFORM_ANIM + 0.2) * 60)):
        b.update(1 / 60)
        for ev in b.drain_events():
            fx.handle_event(ev, c)
        fx.update(1 / 60, c)
        fx.draw_dim(world)
        fx.draw(world, c)
    assert b.transformed and fx.aura_style.name == "gold" and fx.hair_alpha > 0.5
    for name in ("beam_fire", "enemy_hit", "enemy_attack", "player_hit", "shield_hit", "shield_break", "victory",
                 "ultimate_fire", "rage_active", "defeat"):
        fx.handle_event(Event(name, {"aim": 0.5, "hit": True, "charge": 1.0, "damage": 30, "source": "ultimate"}), c)
    for _ in range(90):
        fx.update(1 / 60, c)
        fx.draw(world, c)
    fx.reset()
    assert fx.particles.count == 0 and not fx.beams


def test_aura_style_follows_state():
    b = Battle(rng=random.Random(1))
    fx = Fx("standard", size=(320, 180))
    assert fx._aura_target(b)[0] == "blue"
    b.start_charge()
    assert fx._aura_target(b)[0] == "purple"
    b.fsm.transition(S.BATTLE_IDLE)
    b.rage_time = 5
    assert fx._aura_target(b)[0] == "red"
    b.rage_time = 0
    b.fsm.transition(S.ULTIMATE)
    assert fx._aura_target(b)[0] == "white_gold"
    assert fx._aura_target(None)[1] == 0


def test_energy_depletion_lowers_gold_aura():
    b = Battle(rng=random.Random(1))
    fx = Fx("standard", size=(320, 180))
    b.transformed = True
    b.energy = 100
    hi = fx._aura_target(b)[1]
    b.energy = 5
    assert fx._aura_target(b)[1] < hi


def test_hair_overlay_fallback_and_draw(tmp_path):
    h = HairOverlay(tmp_path / "missing.png")
    assert h.using_fallback
    s = pygame.Surface((400, 300))
    a = PlayerAnchor.from_params((200, 150), 120, 160, roll_deg=15)
    for alpha, tint in ((1.0, None), (0.5, "red"), (1.0, "white"), (0.0, None)):
        h.draw(s, a, alpha, tint)
    h.draw(s, PlayerAnchor.from_params((1e6, -1e6), 1, 1), 1.0)   # degenerate anchor must not crash


def test_hair_overlay_loads_real_asset_when_present():
    if not C.HAIR_PATH.exists():
        pytest.skip("run scripts/setup_assets.py first")
    assert not HairOverlay().using_fallback


def test_procedural_hair_has_transparency():
    surf = make_hair_surface(200, 140)
    assert surf.get_at((0, 0)).a == 0 and surf.get_at((100, 40)).a > 0
