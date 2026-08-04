import numpy as np

from src.utils.kinms_utils import (
    integrated_sb_per_pixel,
    kinms_intflux_from_sb_map,
)


def test_integrated_sb_matches_moment0_convention():
    """Moment-0 = sum_over_channels(Jy/pixel/channel) * z_step_kms."""
    sb_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    z_step_kms = 10.0
    n_channels = 5

    integrated = integrated_sb_per_pixel(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=n_channels,
        input_units="jy_per_pixel_per_channel",
    )
    expected = sb_map * n_channels * z_step_kms
    assert np.allclose(integrated, expected)


def test_kinms_intflux_is_total_jy_per_kms():
    sb_map = np.full((4, 4), 2e-4)
    z_step_kms = 12.5
    n_channels = 8

    intflux = kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=n_channels,
    )
    expected = sb_map.sum() * n_channels * z_step_kms
    assert np.isclose(intflux, expected)


def test_kinms_intflux_accepts_preintegrated_map():
    sb_map = np.full((3, 3), 0.01)
    intflux = kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=10.0,
        n_channels=4,
        sb_input_units="jy_kms_per_pixel",
    )
    assert np.isclose(intflux, sb_map.sum())
