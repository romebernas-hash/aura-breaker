import numpy as np
import pytest

import config as C
from face_controls import (CalibStep, Calibration, Calibrator, FaceInputMapper, FaceMeasurements, Gesture,
                           GestureDetector, measure_landmarks)

DT = 1 / 60
BASE = C.DEFAULT_BASELINE
D = C.DEFAULT_DELTAS


def meas(**kw):
    v = dict(BASE)
    v.update(kw)
    return FaceMeasurements(v["mouth_open"], v["brow_raise"], v["lip_width"], v["roll_deg"])


def mouth_open(k=1.3):
    return meas(mouth_open=BASE["mouth_open"] + D["MOUTH_OPEN"] * k)


def brows_up(k=1.3):
    return meas(brow_raise=BASE["brow_raise"] + D["BROW_RAISE"] * k)


def brows_down(k=1.3):
    return meas(brow_raise=BASE["brow_raise"] + D["BROW_LOWER"] * k)


def purse(k=1.4):
    return meas(lip_width=BASE["lip_width"] + D["PURSE"] * k)


def ready_detector(**kw):
    det = GestureDetector(Calibration(baseline=dict(BASE)), **kw)
    run(det, meas(), 0.5)  # establish neutral
    return det


def run(det, m, seconds, dt=DT):
    frames = []
    for _ in range(int(seconds / dt)):
        frames.append(det.update(m, dt))
    return frames


def pressed(frames):
    return [g for f in frames for g in f.pressed]


# ---------------------------------------------------------------- landmarks
def fake_landmarks(mouth_gap=0.02, roll_dy=0.0):
    lm = np.zeros((478, 2))
    lm[10] = (0.5, 0.2)
    lm[152] = (0.5, 0.8)       # face height 0.6
    lm[234] = (0.3, 0.5)
    lm[454] = (0.7, 0.5)       # face width 0.4
    lm[13] = (0.5, 0.65)
    lm[14] = (0.5, 0.65 + mouth_gap)
    lm[61] = (0.43, 0.67)
    lm[291] = (0.57, 0.67)
    lm[105] = (0.42, 0.38)
    lm[159] = (0.42, 0.44)
    lm[334] = (0.58, 0.38)
    lm[386] = (0.58, 0.44)
    lm[33] = (0.38, 0.45)
    lm[263] = (0.62, 0.45 + roll_dy)
    lm[1] = (0.5, 0.55)
    return lm


def test_measure_landmarks_normalized_values():
    m = measure_landmarks(fake_landmarks(mouth_gap=0.06), aspect=1.0)
    assert m.mouth_open == pytest.approx(0.06 / 0.6)
    assert m.brow_raise == pytest.approx(0.06 / 0.6)
    assert m.lip_width == pytest.approx(0.14 / 0.4)
    assert m.roll_deg == pytest.approx(0.0)


def test_measurements_are_scale_invariant():
    a = measure_landmarks(fake_landmarks())
    lm = fake_landmarks()
    lm[:, :2] = (lm[:, :2] - 0.5) * 0.5 + 0.5  # same face, half size
    b = measure_landmarks(lm)
    assert a.mouth_open == pytest.approx(b.mouth_open)
    assert a.lip_width == pytest.approx(b.lip_width)


def test_roll_sign_left_tilt_positive():
    m = measure_landmarks(fake_landmarks(roll_dy=0.05))
    assert m.roll_deg > 5


@pytest.mark.parametrize("bad", [None, [], np.zeros((10, 3)), np.full((478, 3), np.nan), np.zeros((478, 3))])
def test_invalid_landmarks_return_none(bad):
    assert measure_landmarks(bad) is None


# ---------------------------------------------------------------- thresholds
def test_neutral_triggers_nothing():
    det = ready_detector()
    assert pressed(run(det, meas(), 2.0)) == []


@pytest.mark.parametrize("make,gesture", [(mouth_open, Gesture.MOUTH_OPEN), (brows_up, Gesture.BROW_RAISE),
                                          (brows_down, Gesture.BROW_LOWER), (purse, Gesture.PURSE)])
def test_each_gesture_detected_once(make, gesture):
    det = ready_detector()
    frames = run(det, make(), 1.5)
    assert pressed(frames).count(gesture) == 1
    assert gesture in frames[-1].active


def test_small_movement_below_threshold_ignored():
    det = ready_detector()
    assert pressed(run(det, mouth_open(0.3), 1.0)) == []


def test_hold_time_debounces_short_blips():
    det = ready_detector(hold_time=0.2)
    assert pressed(run(det, mouth_open(), 0.1)) == []          # shorter than hold time
    run(det, meas(), 0.5)
    assert pressed(run(det, mouth_open(), 0.5)) == [Gesture.MOUTH_OPEN]


def test_release_hysteresis_and_event():
    det = ready_detector()
    run(det, mouth_open(), 0.5)
    frames = run(det, mouth_open(0.55), 0.5)   # between OFF and ON thresholds: stays active
    assert Gesture.MOUTH_OPEN in frames[-1].active
    frames = run(det, meas(), 0.5)
    assert Gesture.MOUTH_OPEN not in frames[-1].active
    assert any(Gesture.MOUTH_OPEN in f.released for f in frames)


def test_cooldown_blocks_instant_retrigger_but_allows_later():
    det = ready_detector()
    run(det, mouth_open(), 0.5)
    run(det, meas(), 0.15)  # released, still inside cooldown (0.30 s)
    assert pressed(run(det, mouth_open(), 0.1)) == []
    run(det, meas(), 0.6)
    assert Gesture.MOUTH_OPEN in pressed(run(det, mouth_open(), 0.5))


def test_no_repeat_while_held():
    det = ready_detector()
    assert len(pressed(run(det, mouth_open(), 5.0))) == 1


def test_starting_with_gesture_requires_neutral_first():
    det = GestureDetector(Calibration(baseline=dict(BASE)))
    assert pressed(run(det, mouth_open(), 1.0)) == []         # never neutral -> no trigger
    run(det, meas(), 0.3)
    assert pressed(run(det, mouth_open(), 0.5)) == [Gesture.MOUTH_OPEN]


def test_mutually_exclusive_gestures():
    det = ready_detector()
    both = meas(brow_raise=BASE["brow_raise"] + D["BROW_RAISE"] * 1.3)
    frames = run(det, both, 1.0)
    assert Gesture.BROW_LOWER not in frames[-1].active


def test_sensitivity_scales_scores():
    low = ready_detector(sensitivity=0.5)
    assert pressed(run(low, mouth_open(1.0), 1.0)) == []
    high = ready_detector(sensitivity=1.5)
    assert pressed(run(high, mouth_open(0.6), 1.0)) == [Gesture.MOUTH_OPEN]


def test_face_loss_grace_then_release():
    det = ready_detector()
    run(det, mouth_open(), 0.5)
    frames = run(det, None, 0.3)                       # inside grace period
    assert Gesture.MOUTH_OPEN in frames[-1].active
    frames = run(det, None, 0.6)                       # beyond grace
    assert not frames[-1].face_present
    assert any(Gesture.MOUTH_OPEN in f.released for f in frames)
    # after returning, neutral must be seen again before triggering
    assert pressed(run(det, mouth_open(), 0.5)) == []


def test_auto_baseline_when_uncalibrated():
    det = GestureDetector(Calibration())
    assert not run(det, meas(), 0.5)[-1].tracking_ready
    frames = run(det, meas(), 1.0)
    assert frames[-1].tracking_ready
    run(det, meas(), 0.3)
    assert Gesture.MOUTH_OPEN in pressed(run(det, mouth_open(), 0.6))


def test_aim_from_head_tilt():
    det = ready_detector()
    frames = run(det, meas(roll_deg=12.0), 1.0)
    assert 0.3 < frames[-1].aim <= 1.0
    frames = run(det, meas(roll_deg=-30.0), 2.0)
    assert frames[-1].aim == pytest.approx(-1.0, abs=0.02)
    frames = run(det, meas(roll_deg=1.0), 2.0)          # inside deadzone
    assert abs(frames[-1].aim) < 0.02


def test_zero_dt_and_extreme_values_do_not_crash():
    det = ready_detector()
    det.update(meas(), 0.0)
    det.update(meas(mouth_open=99, brow_raise=-99, lip_width=-5, roll_deg=720), DT)
    assert -1.0 <= det.aim <= 1.0


# ---------------------------------------------------------------- mapper
def test_mapper_brow_lower_is_single_edge_command():
    det, mp = ready_detector(), FaceInputMapper()
    cmds = [mp.map(f, DT) for f in run(det, brows_down(), 2.0)]
    assert sum(c.brow_action for c in cmds) == 1


def test_mapper_charge_and_shield_levels():
    det, mp = ready_detector(), FaceInputMapper()
    last = [mp.map(f, DT) for f in run(det, mouth_open(), 1.0)][-1]
    assert last.charge_held and not last.shield_held
    run(det, meas(), 0.6)
    last = [mp.map(f, DT) for f in run(det, purse(), 1.0)][-1]
    assert last.shield_held and not last.charge_held


def test_mapper_scanner_fires_once_per_hold():
    det, mp = ready_detector(), FaceInputMapper()
    cmds = [mp.map(f, DT) for f in run(det, brows_up(), 2.0)]
    assert sum(c.scan for c in cmds) == 1


def test_mapper_ultimate_needs_combo_hold_and_fires_once():
    det, mp = ready_detector(), FaceInputMapper()
    combo = meas(mouth_open=BASE["mouth_open"] + D["MOUTH_OPEN"] * 1.3,
                 brow_raise=BASE["brow_raise"] + D["BROW_RAISE"] * 1.3)
    short = [mp.map(f, DT) for f in run(det, combo, 0.4)]
    assert not any(c.ultimate for c in short)
    long = [mp.map(f, DT) for f in run(det, combo, 2.0)]
    assert sum(c.ultimate for c in long) == 1
    assert not long[-1].charge_held              # combo suppresses charging


def test_mapper_idle_when_face_missing():
    det, mp = ready_detector(), FaceInputMapper()
    for f in run(det, None, 1.0):
        c = mp.map(f, DT)
    assert not (c.charge_held or c.shield_held or c.scan or c.brow_action or c.ultimate)


# ---------------------------------------------------------------- calibrator
def feed(cal, m, seconds, dt=DT):
    for _ in range(int(seconds / dt)):
        cal.update(m, dt)


def test_calibrator_builds_thresholds():
    cal = Calibrator(prep=0.2, record=0.5)
    sequence = [meas(), mouth_open(2.0), brows_up(2.0), brows_down(2.0), purse(2.0)]
    for m in sequence:
        feed(cal, m, 0.2)   # prep
        feed(cal, m, 0.6)   # record
    assert cal.finished and cal.step == CalibStep.TEST
    built = cal.build()
    assert built.calibrated and not built.warnings
    assert built.deltas["MOUTH_OPEN"] == pytest.approx(D["MOUTH_OPEN"] * 2.0, rel=0.05)
    assert built.deltas["BROW_LOWER"] < 0 and built.deltas["PURSE"] < 0


def test_calibrator_falls_back_when_gesture_too_small():
    cal = Calibrator(prep=0.1, record=0.3)
    for m in [meas(), meas(), meas(), meas(), meas()]:
        feed(cal, m, 0.1)
        feed(cal, m, 0.4)
    built = cal.build()
    assert len(built.warnings) == 4
    assert built.deltas["MOUTH_OPEN"] == D["MOUTH_OPEN"]


def test_calibrator_waits_for_face_during_record():
    cal = Calibrator(prep=0.1, record=0.3)
    feed(cal, None, 0.1)
    feed(cal, None, 2.0)
    assert cal.step == CalibStep.NEUTRAL and cal.phase == "record"
    assert cal.no_face_time > 1.0


def test_calibration_roundtrip_and_bad_data():
    cal = Calibration(baseline=dict(BASE), deltas={"MOUTH_OPEN": 0.05, **{k: v for k, v in D.items() if k != "MOUTH_OPEN"}}, calibrated=True)
    again = Calibration.from_dict(cal.to_dict())
    assert again.deltas["MOUTH_OPEN"] == 0.05 and again.calibrated
    assert not Calibration.from_dict({"baseline": {"x": 1}}).calibrated
    assert not Calibration.from_dict(None).calibrated
