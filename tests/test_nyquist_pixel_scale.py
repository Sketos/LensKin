import numpy as np
import pytest

from src.utils import autolens_utils


def test_nyquist_pixel_scale_from_uv_distance():
    # uv_max = 1e5 wavelengths → λ/b = 1e-5 rad → 0.5 λ/b in arcsec
    uv_max = 1.0e5
    u = np.array([[uv_max, 0.0], [0.0, 0.0]])
    expected = 0.5 / uv_max * (180.0 / np.pi * 3600.0)
    np.testing.assert_allclose(
        autolens_utils.nyquist_pixel_scale_arcsec_from_uv(u),
        expected,
    )


def test_image_plane_grid_nyquist_keeps_n_pixels():
    uv = np.zeros((2, 10, 2))
    uv[0, 0, 0] = 1.0e5
    settings = {
        "n_pixels": 40,
        "real_space_width": 5.0,
        "pixel_scale": "nyquist",
    }
    n, pix, width = autolens_utils.image_plane_grid_from_settings(
        settings, uv_wavelengths=uv
    )
    assert n == 40
    np.testing.assert_allclose(pix, 0.5 / 1.0e5 * (180.0 / np.pi * 3600.0))
    np.testing.assert_allclose(width, 40 * pix)
    # Declared FOV is overridden so the coarse grid stays Nyquist-sampled.
    assert width != 5.0


def test_image_plane_grid_fov_mode_ignores_uv():
    uv = np.zeros((2, 10, 2))
    uv[0, 0, 0] = 1.0e5
    settings = {
        "n_pixels": 40,
        "real_space_width": 5.0,
        "pixel_scale_mode": "fov",
    }
    n, pix, width = autolens_utils.image_plane_grid_from_settings(
        settings, uv_wavelengths=uv
    )
    assert n == 40
    np.testing.assert_allclose(pix, 5.0 / 40.0)
    np.testing.assert_allclose(width, 5.0)


def test_image_plane_grid_explicit_scale():
    settings = {"n_pixels": 40, "real_space_width": 5.0, "pixel_scale": 0.1}
    n, pix, width = autolens_utils.image_plane_grid_from_settings(settings)
    assert n == 40
    np.testing.assert_allclose(pix, 0.1)
    np.testing.assert_allclose(width, 5.0)


def test_nyquist_requires_uv():
    settings = {"n_pixels": 40, "pixel_scale": "nyquist"}
    with pytest.raises(ValueError, match="uv_wavelengths"):
        autolens_utils.image_plane_grid_from_settings(settings)


def test_resolve_image_plane_grid_mutates_settings():
    uv = np.zeros((4, 8, 2))
    uv[0, 0, 1] = 2.0e5
    settings = {"n_pixels": 40, "real_space_width": 5.0, "pixel_scale": "nyquist"}
    autolens_utils.resolve_image_plane_grid_in_settings(settings, uv)
    assert isinstance(settings["pixel_scale"], float)
    np.testing.assert_allclose(
        settings["real_space_width"],
        settings["n_pixels"] * settings["pixel_scale"],
    )
