"""Unit tests for the glove serial parser, calibration and mock backend.

These tests do **not** require any hardware — everything runs against the
``MOCK`` backend or in-memory fixtures.

The firmware was downsized to **3 flex sensors** (thumb / index / middle)
plus the MPU6050 IMU, after one of the original 5 flex sensors lost its
solder pad. The CSV record is therefore 10 fields:

    t_ms,f1,f2,f3,ax,ay,az,gx,gy,gz
"""

from __future__ import annotations

import time

import pytest

from glove_reader import (
    FEATURES_PER_FRAME,
    NUM_FLEX,
    GloveCalibration,
    GloveFrame,
    GloveReader,
    parse_csv_line,
)


# ── parser ───────────────────────────────────────────────────────────────────

class TestParseCsvLine:
    def test_well_formed_line(self):
        line = "1234,2100,2200,2300,0.05,-0.10,0.99,1.5,-2.0,0.7"
        f = parse_csv_line(line)
        assert f is not None
        assert f.t_ms == 1234
        assert f.flex == (2100, 2200, 2300)
        assert f.accel == pytest.approx((0.05, -0.10, 0.99))
        assert f.gyro == pytest.approx((1.5, -2.0, 0.7))

    def test_strips_trailing_whitespace_and_cr(self):
        line = "  1,1,2,3,0,0,1,0,0,0  \r"
        assert parse_csv_line(line) is not None

    @pytest.mark.parametrize("bad", [
        "",
        "\n",
        "# comment line",
        "# fmt: t,f1,f2",
        "not enough,fields",
        "1,2,3,4,5,6,7,8,9",                    # 9 fields
        "1,2,3,4,5,6,7,8,9,10,11",              # 11 fields
        "abc,2100,2200,2300,0,0,1,0,0,0",       # bad timestamp
        "1,a,b,c,0,0,1,0,0,0",                  # bad flex ints
        "1,1,2,3,x,y,z,0,0,0",                  # bad floats
    ])
    def test_malformed_returns_none(self, bad):
        assert parse_csv_line(bad) is None


# ── calibration ──────────────────────────────────────────────────────────────

class TestGloveCalibration:
    def test_default_normalises_in_unit_range(self):
        c = GloveCalibration()
        f = GloveFrame(t_ms=0, flex=(1500, 2250, 3000),
                       accel=(0.0, 0.0, 1.0), gyro=(0.0, 0.0, 0.0))
        feats = c.normalise(f)
        assert len(feats) == FEATURES_PER_FRAME
        assert feats[0] == pytest.approx(0.0)
        assert feats[1] == pytest.approx(0.5)
        assert feats[2] == pytest.approx(1.0)

    def test_clamps_out_of_range(self):
        c = GloveCalibration()
        f = GloveFrame(t_ms=0, flex=(0, 4095, 2250),
                       accel=(0, 0, 0), gyro=(0, 0, 0))
        feats = c.normalise(f)
        assert feats[0] == 0.0
        assert feats[1] == 1.0

    def test_gyro_scaled_to_unit(self):
        c = GloveCalibration()
        f = GloveFrame(t_ms=0, flex=(1500,) * NUM_FLEX,
                       accel=(0, 0, 0), gyro=(250, -250, 0))
        feats = c.normalise(f)
        # feats layout: 3 flex + ax,ay,az + gx,gy,gz  -> indices 6, 7, 8
        assert feats[6] == pytest.approx(1.0)   # gx / 250
        assert feats[7] == pytest.approx(-1.0)  # gy / 250

    def test_from_samples_finds_min_and_max(self):
        frames = [
            GloveFrame(0, (1000, 1100, 1200), (0, 0, 1), (0, 0, 0)),
            GloveFrame(1, (3500, 3400, 3300), (0, 0, 1), (0, 0, 0)),
            GloveFrame(2, (2000, 2000, 2000), (0, 0, 1), (0, 0, 0)),
        ]
        c = GloveCalibration.from_samples(frames)
        assert c.flex_min == (1000, 1100, 1200)
        assert c.flex_max == (3500, 3400, 3300)

    def test_from_samples_avoids_zero_range(self):
        flat = [GloveFrame(i, (1000,) * NUM_FLEX, (0, 0, 1), (0, 0, 0))
                for i in range(20)]
        c = GloveCalibration.from_samples(flat)
        for lo, hi in zip(c.flex_min, c.flex_max):
            assert hi - lo >= 100

    def test_round_trip_serialisation(self):
        c1 = GloveCalibration(flex_min=(100, 200, 300),
                              flex_max=(900, 1000, 1100))
        d = c1.to_dict()
        c2 = GloveCalibration.from_dict(d)
        assert c2.flex_min == c1.flex_min
        assert c2.flex_max == c1.flex_max

    def test_legacy_5_flex_calibration_truncated(self):
        """Old calibration files (5 flex) must be loadable in the 3-flex build."""
        legacy = {
            "flex_min": [1500, 1600, 1700, 1800, 1900],
            "flex_max": [3000, 3100, 3200, 3300, 3400],
        }
        c = GloveCalibration.from_dict(legacy)
        assert c.flex_min == (1500, 1600, 1700)
        assert c.flex_max == (3000, 3100, 3200)


# ── mock backend ─────────────────────────────────────────────────────────────

class TestMockReader:
    def test_produces_frames(self):
        reader = GloveReader(port="MOCK")
        assert reader.start()
        try:
            time.sleep(0.3)  # ~15 frames at 50 Hz
            f = reader.read_latest()
            assert f is not None
            assert isinstance(f, GloveFrame)
            assert len(f.flex) == NUM_FLEX
            assert reader.frames_seen >= 5
        finally:
            reader.stop()

    def test_drain_returns_in_order(self):
        reader = GloveReader(port="MOCK", buffer_size=64)
        assert reader.start()
        try:
            time.sleep(0.4)
            frames = reader.drain()
            assert len(frames) > 0
            ts = [f.t_ms for f in frames]
            assert ts == sorted(ts)
        finally:
            reader.stop()

    def test_context_manager(self):
        with GloveReader(port="MOCK") as r:
            time.sleep(0.1)
            assert r.read_latest() is not None
        assert r._thread is None

    def test_push_is_threadsafe_for_tests(self):
        reader = GloveReader(port="MOCK")
        # don't actually start the mock loop — push directly
        f = GloveFrame(t_ms=42, flex=(1, 2, 3),
                       accel=(0, 0, 1), gyro=(0, 0, 0))
        reader.push(f)
        assert reader.read_latest() is f
        assert reader.frames_seen == 1
