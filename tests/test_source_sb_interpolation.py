"""Tests for phase-1 reconstruction → KinMS source-grid interpolation."""
import numpy as np
import pytest

from src.grid.grid import Grid3D
from src.pipelines.reconstruction import (
    interpolate_reconstruction_to_grid,
    rescale_sb_from_image_pixels,
)
from src.utils import kinms_utils


def _uniform_mesh_coords(n, width=5.0):
    """(n*n, 2) autolens-style (y, x) centres covering ``[-width/2, width/2]``."""
    pixel = width / n
    edges = np.linspace(-width / 2 + pixel / 2, width / 2 - pixel / 2, n)
    yy, xx = np.meshgrid(edges, edges, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def test_finer_grid_sum_scales_with_pixel_area_ratio():
    """
    Constant SB on a 20² mesh interpolated to 40² must grow the sum by ~4.

    Raw interpolation stays in image-pixel Jy units; the sum growth is expected
    before :func:`rescale_sb_from_image_pixels`. Edge fill outside the mesh
    convex hull makes the ratio slightly below 4.
    """
    width = 5.0
    mesh_n = 20
    dest_n = 40
    brightness = 0.01

    mesh = _uniform_mesh_coords(mesh_n, width=width)
    recon = np.full(mesh.shape[0], brightness)
    dest = Grid3D.uniform(n_pixels=dest_n, pixel_scale=width / dest_n, n_channels=2)

    sb_map = interpolate_reconstruction_to_grid(recon, mesh, dest.grid_2d)

    area_ratio = (dest_n / mesh_n) ** 2
    sum_ratio = float(sb_map.sum() / recon.sum())
    assert sb_map.shape == (dest_n, dest_n)
    # Must NOT preserve the mesh sum (old bug → ratio ≈ 1).
    assert sum_ratio > 3.0
    assert sum_ratio == pytest.approx(area_ratio, rel=0.15)
    assert float(np.nanmedian(sb_map[sb_map != 0])) == pytest.approx(brightness, rel=0.05)


def test_interpolated_sb_intflux_scales_with_area_not_mesh_sum():
    width = 5.0
    mesh_n = 20
    dest_n = 40
    n_channels = 16
    z_step = 30.0
    brightness = 0.01
    image_pixel_scale = width / dest_n  # dest matches image pixel scale

    mesh = _uniform_mesh_coords(mesh_n, width=width)
    recon = np.full(mesh.shape[0], brightness)
    dest = Grid3D.uniform(
        n_pixels=dest_n, pixel_scale=image_pixel_scale, n_channels=n_channels
    )
    sb_image_units = interpolate_reconstruction_to_grid(recon, mesh, dest.grid_2d)
    sb_map = rescale_sb_from_image_pixels(
        sb_image_units,
        image_pixel_scale=image_pixel_scale,
        grid_pixel_scale=dest.pixel_scale,
    )

    intflux = kinms_utils.kinms_intflux_from_sb_map(
        sb_map, z_step_kms=z_step, n_channels=n_channels
    )
    # Wrong (old) convention: preserve mesh sum → undercounts by area_ratio.
    wrong = recon.sum() * n_channels * z_step
    assert intflux / wrong > 3.0
    assert intflux / wrong == pytest.approx((dest_n / mesh_n) ** 2, rel=0.15)


def test_rescale_sb_converts_image_pixel_units_to_finer_grid():
    """Finer-than-image grids must reduce Jy/pixel so total flux is conserved."""
    sb_image_units = np.ones((4, 4))
    image_ps = 0.16
    grid_ps = 0.08
    sb = rescale_sb_from_image_pixels(sb_image_units, image_ps, grid_ps)
    assert sb.sum() == pytest.approx(sb_image_units.sum() * (grid_ps / image_ps) ** 2)
    assert sb[0, 0] == pytest.approx(0.25)
