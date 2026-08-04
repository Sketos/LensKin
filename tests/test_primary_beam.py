"""Tests for the primary-beam attenuation module."""
import numpy as np
import pytest

from src.utils.primary_beam import (
    SPEED_OF_LIGHT_M_S,
    hpbw_arcsec,
    primary_beam_image,
    primary_beam_from_settings,
)


class TestHPBW:
    def test_band7_alma(self):
        """HPBW at 350 GHz, 12-m dish should be ~17 arcsec."""
        freq_hz = 350e9
        hpbw = hpbw_arcsec(freq_hz, dish_diameter_m=12.0)
        expected = np.degrees(1.13 * (SPEED_OF_LIGHT_M_S / freq_hz) / 12.0) * 3600.0
        assert abs(hpbw - expected) < 1e-10
        assert 15.0 < hpbw < 20.0

    def test_scales_with_frequency(self):
        """Lower frequency → larger beam."""
        assert hpbw_arcsec(100e9) > hpbw_arcsec(350e9)

    def test_scales_with_dish(self):
        """Larger dish → smaller beam."""
        assert hpbw_arcsec(350e9, dish_diameter_m=25.0) < hpbw_arcsec(350e9, dish_diameter_m=12.0)


class TestPrimaryBeamImage:
    @pytest.fixture
    def grid_5arcsec(self):
        """10×10 grid spanning ±2.5 arcsec, last axis = (y, x)."""
        y, x = np.mgrid[-2.5:2.5:10j, -2.5:2.5:10j]
        return np.stack([y, x], axis=-1)

    def test_peak_at_centre(self, grid_5arcsec):
        pb = primary_beam_image(grid_5arcsec, frequency_hz=350e9)
        centre_idx = (5, 5)
        assert abs(pb[centre_idx] - 1.0) < 0.05

    def test_monotonic_decrease(self, grid_5arcsec):
        pb = primary_beam_image(grid_5arcsec, frequency_hz=350e9)
        centre = pb[5, 5]
        corner = pb[0, 0]
        assert corner < centre

    def test_values_between_0_and_1(self, grid_5arcsec):
        pb = primary_beam_image(grid_5arcsec, frequency_hz=350e9)
        assert pb.min() >= 0.0
        assert pb.max() <= 1.0

    def test_offset_pointing(self, grid_5arcsec):
        """Peak should shift when pointing is offset."""
        pb = primary_beam_image(
            grid_5arcsec, frequency_hz=350e9, pointing_arcsec=(1.0, 1.0)
        )
        centre_val = pb[5, 5]
        idx_near_offset = (7, 7)
        assert pb[idx_near_offset] > centre_val

    def test_shape_matches_grid(self, grid_5arcsec):
        pb = primary_beam_image(grid_5arcsec, frequency_hz=350e9)
        assert pb.shape == (10, 10)

    def test_band7_attenuation_at_2arcsec(self):
        """At ~2 arcsec from centre, Band 7 PB should be >0.95."""
        grid = np.array([[[0.0, 0.0], [0.0, 2.0]]])
        pb = primary_beam_image(grid, frequency_hz=350e9)
        assert pb[0, 1] > 0.95


class TestPrimaryBeamFromSettings:
    def test_disabled_returns_none(self):
        settings = {"primary_beam": {"enabled": False}}
        result = primary_beam_from_settings(settings, np.array([350e9]), mock_mask())
        assert result is None

    def test_absent_returns_none(self):
        result = primary_beam_from_settings({}, np.array([350e9]), mock_mask())
        assert result is None

    def test_enabled_returns_array(self):
        settings = {
            "primary_beam": {"enabled": True, "dish_diameter_m": 12.0},
        }
        mask = mock_mask()
        pb = primary_beam_from_settings(settings, np.array([350e9]), mask)
        assert pb is not None
        assert pb.shape == (10, 10)
        assert 0.0 < pb.min() <= pb.max() <= 1.0

    def test_uses_mean_frequency(self):
        settings = {"primary_beam": {"enabled": True}}
        freqs = np.array([340e9, 360e9])
        mask = mock_mask()
        pb = primary_beam_from_settings(settings, freqs, mask)
        pb_single = primary_beam_from_settings(settings, np.array([350e9]), mask)
        np.testing.assert_allclose(pb, pb_single)


class TestVisibilitiesWithPB:
    """Test that primary_beam kwarg in visibilities_from_transformers_and_cube works."""

    def test_pb_none_is_identity(self):
        """When primary_beam=None, output should be identical to the default."""
        cube = np.random.randn(2, 10, 10)
        shape = (2, 5, 2)
        t1, t2 = MockTransformer(), MockTransformer()
        from src.utils.autolens_utils import visibilities_from_transformers_and_cube

        vis_no_pb = visibilities_from_transformers_and_cube(
            cube, [t1, t2], shape, primary_beam=None
        )
        vis_default = visibilities_from_transformers_and_cube(
            cube, [t1, t2], shape
        )
        np.testing.assert_array_equal(vis_no_pb, vis_default)

    def test_pb_ones_is_identity(self):
        """PB of all ones should not change the visibilities."""
        cube = np.random.randn(2, 10, 10)
        shape = (2, 5, 2)
        t1, t2 = MockTransformer(), MockTransformer()
        from src.utils.autolens_utils import visibilities_from_transformers_and_cube

        pb = np.ones((10, 10))
        vis_pb = visibilities_from_transformers_and_cube(
            cube, [t1, t2], shape, primary_beam=pb
        )
        vis_no_pb = visibilities_from_transformers_and_cube(
            cube, [t1, t2], shape
        )
        np.testing.assert_array_equal(vis_pb, vis_no_pb)

    def test_pb_attenuates(self):
        """PB < 1 should reduce the visibility amplitudes."""
        cube = np.ones((1, 10, 10))
        shape = (1, 5, 2)
        t = MockTransformer()
        from src.utils.autolens_utils import visibilities_from_transformers_and_cube

        vis_no_pb = visibilities_from_transformers_and_cube(cube, [t], shape)
        pb = 0.5 * np.ones((10, 10))
        vis_pb = visibilities_from_transformers_and_cube(cube, [t], shape, primary_beam=pb)
        np.testing.assert_allclose(vis_pb, 0.5 * vis_no_pb)


# ── Helpers ──────────────────────────────────────────────────────────────────


def mock_mask():
    """Minimal object that mimics mask_2d.derive_grid.all_false.native."""

    class _Grid:
        def __init__(self):
            y, x = np.mgrid[-2.5:2.5:10j, -2.5:2.5:10j]
            self._native = np.stack([y, x], axis=-1)

        @property
        def native(self):
            return self._native

    class _Derive:
        def __init__(self):
            self.all_false = _Grid()

    class _Mask:
        def __init__(self):
            self.derive_grid = _Derive()

    return _Mask()


class _FakeArray2D(np.ndarray):
    """Minimal stand-in for al.Array2D used by the transformer mock."""

    def __new__(cls, values, mask=None):
        obj = np.asarray(values).view(cls)
        obj.mask = mask
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mask = getattr(obj, "mask", None)


class _FakeVisibilities:
    def __init__(self, arr):
        self.array = arr


class MockTransformer:
    """Minimal transformer mock for unit-testing the PB multiply path."""

    def __init__(self):
        self.real_space_mask = type("M", (), {"shape_native": (10, 10)})()
        self._n_vis = 5

    def visibilities_from(self, image, **kw):
        flat = np.asarray(image).ravel()
        vis_r = flat[: self._n_vis]
        vis_i = flat[self._n_vis : 2 * self._n_vis]
        return _FakeVisibilities(vis_r + 1j * vis_i)


# Monkey-patch al.Array2D so autolens_utils can construct it without the real import
import src.utils.autolens_utils as _au
_au.al = type("_al", (), {"Array2D": _FakeArray2D})()
