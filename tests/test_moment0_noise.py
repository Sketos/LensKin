import numpy as np

from src.pipelines.moment0_noise import (
    collapse_visibilities_to_moment0,
    moment0_arrays_from,
)


def test_independent_mean_equal_sigmas():
    n_channels, n_vis = 8, 4
    sigma_value = 0.1
    visibilities = np.ones((n_channels, n_vis, 2))
    sigma = np.full((n_channels, n_vis, 2), sigma_value)

    _, sigma_mom0 = collapse_visibilities_to_moment0(
        visibilities=visibilities,
        sigma=sigma,
        sigma_mode="independent_mean",
    )

    expected = sigma_value / np.sqrt(n_channels)
    assert np.allclose(sigma_mom0, expected)


def test_sigma_scale_applied():
    visibilities = np.zeros((4, 2, 2))
    sigma = np.ones((4, 2, 2))

    _, sigma_mom0 = collapse_visibilities_to_moment0(
        visibilities=visibilities,
        sigma=sigma,
        sigma_scale=0.5,
    )

    assert np.allclose(sigma_mom0, 0.5 * np.sqrt(4) / 4)


def test_weights_mode_matches_independent_mean_for_equal_weights():
    n_channels, n_vis = 4, 3
    weight = np.full((n_channels, n_vis), 100.0)
    visibilities = np.zeros((n_channels, n_vis, 2))

    _, sigma_weights = collapse_visibilities_to_moment0(
        visibilities=visibilities,
        sigma=None,
        sigma_mode="weights",
        weights=weight,
    )

    sigma = np.full((n_channels, n_vis, 2), 0.1)
    _, sigma_statwt = collapse_visibilities_to_moment0(
        visibilities=visibilities,
        sigma=sigma,
        sigma_mode="independent_mean",
    )

    assert np.allclose(sigma_weights, sigma_statwt)


def test_concatenate_preserves_per_channel_sigma():
    n_channels, n_vis = 3, 4
    visibilities = np.arange(n_channels * n_vis * 2, dtype=float).reshape(
        n_channels, n_vis, 2
    )
    sigma = np.linspace(0.1, 0.5, n_channels * n_vis * 2).reshape(
        n_channels, n_vis, 2
    )
    uv = np.arange(n_channels * n_vis * 2, dtype=float).reshape(n_channels, n_vis, 2)

    vis_out, sigma_out, uv_out = moment0_arrays_from(
        uv_wavelengths=uv,
        visibilities=visibilities,
        sigma=sigma,
        moment0_settings={"collapse_mode": "concatenate"},
    )

    assert vis_out.shape == (n_channels * n_vis,)
    assert sigma_out.shape == (n_channels * n_vis, 2)
    assert uv_out.shape == (n_channels * n_vis, 2)
    assert np.allclose(sigma_out, sigma.reshape(-1, 2))
    assert np.allclose(uv_out, uv.reshape(-1, 2))


def test_concatenate_matches_single_channel_subset():
    n_channels, n_vis = 2, 3
    visibilities = np.ones((n_channels, n_vis, 2))
    visibilities[1, :, 0] = 2.0
    sigma = np.full((n_channels, n_vis, 2), 0.2)
    uv = np.zeros((n_channels, n_vis, 2))
    uv[1, :, 0] = 5.0

    vis_out, _, uv_out = moment0_arrays_from(
        uv_wavelengths=uv,
        visibilities=visibilities,
        sigma=sigma,
        moment0_settings={"collapse_mode": "concatenate"},
    )

    vis_complex = visibilities[..., 0] + 1j * visibilities[..., 1]
    assert np.allclose(vis_out[n_vis:], vis_complex[1])
    assert np.allclose(uv_out[n_vis:], uv[1])
