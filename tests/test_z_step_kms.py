import numpy as np
from pathlib import Path

from src.utils import spectral_utils
from src.utils.kinms_utils import kinms_radial_grid


def _mock_frequencies_hz():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/kinms_mock_pixelized/frequencies_kinms_mock_width_30kms_contsub.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"mock frequencies not found: {path}")
    return np.load(path)


def test_z_step_matches_mock_channel_width():
    frequencies_hz = _mock_frequencies_hz()
    z_step = spectral_utils.z_step_kms_from_data_frequencies(frequencies_hz)
    np.testing.assert_allclose(z_step, 30.0, rtol=0.02)


def test_rest_line_reference_overestimates_z_step_for_mock_ms():
    from astropy import units as u

    frequencies_hz = _mock_frequencies_hz()
    z_correct = spectral_utils.z_step_kms_from_data_frequencies(frequencies_hz)

    frequency_0 = spectral_utils.observed_line_frequency_from_rest_line_frequency(
        frequency=1036.912393,
        redshift=2.7855,
    )
    velocities = spectral_utils.convert_frequencies_to_velocities(
        frequencies=frequencies_hz * u.Hz.to(u.GHz),
        frequency_0=frequency_0,
        units=u.km / u.s,
    )
    z_wrong = abs(velocities[1] - velocities[0])

    assert z_wrong > z_correct * 1.25
    np.testing.assert_allclose(z_wrong, 38.3, rtol=0.02)


def test_kinms_radial_grid_matches_mock_generator():
    pixel_scale = 5.0 / 40.0
    field_width = 5.0
    x = kinms_radial_grid(pixel_scale, field_width)

    log_xmin = np.log10(pixel_scale / 5.0)
    log_xmax = np.log10(field_width)
    expected = np.logspace(log_xmin, log_xmax, 10000)

    np.testing.assert_allclose(x, expected)
    np.testing.assert_allclose(x[-1], field_width)
