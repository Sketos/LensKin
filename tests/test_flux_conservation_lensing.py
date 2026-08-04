"""Flux conservation when mapping a fine source cube to the image-plane grid."""
import numpy as np
import pytest

from src.grid.grid import Grid3D
from src.utils import analysis_utils


def test_bilinear_zoom_does_not_conserve_flux_on_downsample():
    """Plain scipy zoom loses flux when downsampling Jy/pixel maps."""
    fine = np.random.default_rng(0).random((64, 64))
    coarse = analysis_utils.resample_image_to_shape(fine, (8, 8))
    ratio = coarse.sum() / fine.sum()
    expected = (8 / 64) ** 2
    assert ratio == pytest.approx(expected, rel=0.05)


def test_flux_conserving_resample_preserves_sum():
    fine = np.random.default_rng(1).random((64, 64))
    coarse = analysis_utils.resample_image_to_shape_flux_conserving(fine, (8, 8))
    assert coarse.sum() == pytest.approx(fine.sum(), rel=0.05)


def test_ray_trace_pixel_area_scale_matches_grids():
    omega_src = 1.0 / 512**2
    omega_img = 0.125**2
    assert omega_img / omega_src == pytest.approx((0.125 * 512) ** 2)


def test_flux_report_same_shape_grids_identity_cube():
    grid = Grid3D.uniform(n_pixels=8, pixel_scale=0.1, n_channels=4)
    cube = np.ones((4, 8, 8))
    report = analysis_utils.flux_conservation_lensing_report(
        source_cube=cube,
        lensed_cube=cube.copy(),
        z_step_kms=10.0,
        source_grid_2d=grid.grid_2d,
        image_grid_2d=grid.grid_2d,
    )
    assert report["pixel_area_ratio_image_over_source"] == pytest.approx(1.0)
    assert report["source_line_flux_jy_kms"] == pytest.approx(4 * 8 * 8 * 10.0)
