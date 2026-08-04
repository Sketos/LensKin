import numpy as np

from src.grid.grid import Grid3D
from src.utils import autolens_utils


def test_kinms_source_n_pixels_default():
    settings = {"n_pixels": 40, "real_space_width": 5.0}
    assert autolens_utils.kinms_source_n_pixels_from_settings(settings) == 256


def test_kinms_source_n_pixels_from_source_grid():
    settings = {
        "n_pixels": 40,
        "real_space_width": 5.0,
        "source_grid": {"n_pixels": 512},
    }
    assert autolens_utils.kinms_source_n_pixels_from_settings(settings) == 512


def test_kinms_source_uniform_grid_defaults_to_256():
    settings = {"n_pixels": 40, "real_space_width": 5.0}
    n_pixels, pixel_scale, width = autolens_utils.source_grid_from_settings(settings)
    assert n_pixels == 256
    assert width == 5.0
    np.testing.assert_allclose(pixel_scale, 5.0 / 256)


def test_kinms_source_grid_bounding_box_from_settings():
    settings = {
        "n_pixels": 40,
        "real_space_width": 5.0,
        "source_grid": {
            "xmin": -0.5,
            "xmax": 0.5,
            "ymin": -0.5,
            "ymax": 0.5,
            "n_pixels": 512,
        },
    }
    grid = autolens_utils.kinms_source_grid_3d_from_settings(settings, n_channels=16)
    assert isinstance(grid, Grid3D)
    assert grid.n_pixels == 512
    assert grid.n_channels == 16

    coords = np.asarray(grid.grid_2d)
    np.testing.assert_allclose(coords[:, 1].min(), -0.5, atol=0.01)
    np.testing.assert_allclose(coords[:, 1].max(), 0.5, atol=0.01)
    np.testing.assert_allclose(coords[:, 0].min(), -0.5, atol=0.01)
    np.testing.assert_allclose(coords[:, 0].max(), 0.5, atol=0.01)

    label = autolens_utils.source_grid_label_from_settings(settings)
    assert "512" in label
    assert "-0.5" in label
