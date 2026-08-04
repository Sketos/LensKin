"""Pixelized KinMS cloud placement: centre-relative, max-radius, sub-sampling."""

import numpy as np

from src.grid.grid import Grid3D
from src.model import profiles
from src.utils import kinms_utils


def _compact_sb_map(grid_3d, centre, width=0.04):
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    y = coords[:, :, 0]
    x = coords[:, :, 1]
    return np.exp(
        -0.5 * (((x - centre[0]) / width) ** 2 + ((y - centre[1]) / width) ** 2)
    )


def test_pixelized_inclouds_are_recentred_onto_phase_centre():
    n_pixels = 32
    n_channels = 8
    z_step = 30.0
    centre = (0.15, -0.10)

    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=0.05,
        n_channels=n_channels,
    )
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    sb_map = _compact_sb_map(grid_3d, centre)

    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step,
        sb_map=sb_map,
        sb_input_units="jy_per_pixel_per_channel",
        max_radius=1.0,
        clouds_per_pixel=4,
        scale_height_arcsec=0.01,
    )
    assert np.isclose(
        np.average(instance.inClouds[:, 0], weights=instance.flux_clouds),
        centre[0],
        atol=0.03,
    )
    assert np.isclose(
        np.average(instance.inClouds[:, 1], weights=instance.flux_clouds),
        centre[1],
        atol=0.03,
    )
    assert np.std(instance.inClouds[:, 2]) > 0.0

    profile = profiles.kinMSPixelized(
        centre=centre,
        z_centre=0.0,
        inclination=60.0,
        phi=90.0,
        turnover_radius=0.05,
        maximum_velocity=200.0,
        velocity_dispersion=20.0,
        vmax_black_hole=0.0,
    )
    cube = profile.profile_cube_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step,
        instance=instance,
    )
    # Apply the same KinMS y-flip used before ray-tracing.
    mom0 = cube[:, ::-1, :].sum(axis=0) * z_step
    peak_i, peak_j = np.unravel_index(np.nanargmax(mom0), mom0.shape)
    peak_y = float(coords[peak_i, peak_j, 0])
    peak_x = float(coords[peak_i, peak_j, 1])
    assert abs(peak_x - centre[0]) < 0.08
    assert abs(peak_y - centre[1]) < 0.08


def test_pixelized_max_radius_cuts_clouds_about_centre():
    n_pixels = 40
    n_channels = 4
    z_step = 30.0
    centre = (0.0, 0.0)
    max_radius = 0.2

    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=0.05,
        n_channels=n_channels,
    )
    sb_map = _compact_sb_map(grid_3d, centre, width=0.25)
    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step,
        sb_map=sb_map,
        sb_input_units="jy_per_pixel_per_channel",
        max_radius=max_radius,
        clouds_per_pixel=1,
        scale_height_arcsec=0.0,
    )
    assert instance.max_radius == max_radius

    relative = np.asarray(instance.inClouds, dtype=float).copy()
    relative[:, 0] -= centre[0]
    relative[:, 1] -= centre[1]
    kept, kept_flux, kept_int = kinms_utils.apply_max_radius_to_clouds(
        in_clouds=relative,
        flux_clouds=instance.flux_clouds,
        int_flux=instance.int_flux,
        max_radius=max_radius,
    )
    assert kept.shape[0] < instance.inClouds.shape[0]
    assert np.all(np.hypot(kept[:, 0], kept[:, 1]) <= max_radius + 1e-12)
    assert np.isclose(kept_flux.sum(), 1.0)
    assert kept_int < instance.int_flux


def test_clouds_per_pixel_subsamples_and_scale_height_from_redshift():
    settings = {
        "redshift_source": 2.7855,
        "disk_scale_height_kpc": 0.1,
        "reconstruction": {
            "clouds_per_pixel": 9,
            "max_radius": 1.0,
        },
    }
    h_arcsec = kinms_utils.disk_scale_height_arcsec_from_settings(settings)
    assert 0.01 < h_arcsec < 0.05  # ~0.1 kpc at z~2.8 is a few 0.01"

    grid_3d = Grid3D.uniform(n_pixels=20, pixel_scale=0.05, n_channels=4)
    sb_map = _compact_sb_map(grid_3d, (0.0, 0.0), width=0.08)
    n_pix = int(np.count_nonzero(sb_map > 0.0))
    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=30.0,
        sb_map=sb_map,
        **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
    )
    assert instance.inClouds.shape[0] == n_pix * 9
    assert instance.disk_thick == h_arcsec
    assert np.std(instance.inClouds[:, 2]) > 0.0


def test_sky_plane_vlos_matches_kinms_sign_convention():
    """PA=0: cloud on +y (approaching/receding sense via KinMS −cosθ sin i)."""
    vel_rad = np.linspace(0.0, 2.0, 50)
    vel_prof = np.full_like(vel_rad, 200.0)
    # Disk-plane point on +x → θ=0 → v_los = -v sin(i)
    v = kinms_utils.sky_plane_vlos_kms(
        x_arcsec=np.array([1.0]),
        y_arcsec=np.array([0.0]),
        vel_rad_arcsec=vel_rad,
        vel_prof_kms=vel_prof,
        inclination_deg=90.0,
        pos_ang_deg=90.0,  # KinMS identity PA mapping (x,y)→(x,y)
        gas_sigma_kms=0.0,
    )
    assert np.isclose(v[0], -200.0, atol=1.0)


def test_pixelized_inclined_does_not_reproject_morphology():
    """Sky-plane SB must not be compressed by a second inclination."""
    n_pixels = 64
    n_channels = 8
    z_step = 30.0
    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels, pixel_scale=0.05, n_channels=n_channels
    )
    sb_map = _compact_sb_map(grid_3d, (0.0, 0.0), width=0.2)
    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step,
        sb_map=sb_map,
        sb_input_units="jy_per_pixel_per_channel",
        max_radius=None,
        clouds_per_pixel=16,
        scale_height_arcsec=0.0,
        seed=100,
    )
    # Match KinMS cell size to Autolens pixel scale.
    pixel_scale = float(grid_3d.pixel_scales[1])
    instance.obj = kinms_utils.make_kinms_instance(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=n_channels,
        z_step_kms=z_step,
    )
    # Keep |v_los| inside the cube (± n_channels/2 * dv) so cleanOut does not
    # redistribute clipped flux into fewer spatial pixels.
    profile = profiles.kinMSPixelized(
        centre=(0.0, 0.0),
        inclination=65.0,
        phi=90.0,
        turnover_radius=0.05,
        maximum_velocity=50.0,
        velocity_dispersion=0.0,
    )
    cube = profile.profile_cube_from_grid(grid_3d, z_step, instance=instance)
    mom0 = cube.sum(axis=0) * z_step
    sb_mom0 = sb_map * n_channels * z_step
    peak_ratio = float(mom0.max() / sb_mom0.max())
    # Old double-projection path gave ~1/cos(65°) ≈ 2.4; sky-plane path ~1.
    assert peak_ratio < 1.5, f"unexpected brightening: peak_ratio={peak_ratio:.3f}"
    assert np.isclose(mom0.sum(), sb_mom0.sum(), rtol=1e-6)


def test_kinms_cube_axes_match_autolens_grid_yx():
    """
    After KinMS (x,y,v) → (v,y,x), a compact SB peak at (x, y) matches the
    autolens grid once the same ``flip_kinms_y`` as lensing is applied.
    """
    n_pixels = 64
    n_channels = 8
    z_step = 30.0
    pixel_scale = 0.05
    centre = (0.5, -0.25)  # (x, y) arcsec
    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels, pixel_scale=pixel_scale, n_channels=n_channels
    )
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    sb_map = _compact_sb_map(grid_3d, centre, width=0.04)
    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step,
        sb_map=sb_map,
        sb_input_units="jy_per_pixel_per_channel",
        max_radius=None,
        clouds_per_pixel=8,
        scale_height_arcsec=0.0,
        seed=100,
    )
    profile = profiles.kinMSPixelized(
        centre=centre,
        inclination=0.0,
        phi=0.0,
        turnover_radius=0.05,
        maximum_velocity=50.0,
        velocity_dispersion=0.0,
    )
    cube = profile.profile_cube_from_grid(grid_3d, z_step, instance=instance)
    # Match Analysis default: flip KinMS y before comparing to autolens Grid2D.
    mom0 = cube[:, ::-1, :].sum(axis=0)
    peak_i, peak_j = np.unravel_index(np.nanargmax(mom0), mom0.shape)
    peak_y = float(coords[peak_i, peak_j, 0])
    peak_x = float(coords[peak_i, peak_j, 1])
    assert abs(peak_x - centre[0]) < 2.0 * pixel_scale
    assert abs(peak_y - centre[1]) < 2.0 * pixel_scale


def test_max_radius_default_from_settings_is_none():
    assert kinms_utils.max_radius_from_settings({}) is None
    assert kinms_utils.max_radius_from_settings({"reconstruction": {}}) is None
    assert (
        kinms_utils.max_radius_from_settings({"reconstruction": {"max_radius": None}})
        is None
    )
    assert (
        kinms_utils.max_radius_from_settings({"reconstruction": {"max_radius": 1.5}})
        == 1.5
    )
