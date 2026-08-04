"""Tests for dirty-image noise propagation helpers."""

import numpy as np

from src.utils.autolens_utils import dirty_mom0_noise_from_channel_noise


def test_dirty_mom0_noise_from_independent_channels():
    # Two channels, uniform σ=2 and σ=3 → mom0 σ = dv * sqrt(4+9)
    channel_noise = np.ones((2, 4, 4), dtype=float)
    channel_noise[0] *= 2.0
    channel_noise[1] *= 3.0
    dv = 10.0
    mom0 = dirty_mom0_noise_from_channel_noise(channel_noise, dv)
    assert mom0.shape == (4, 4)
    expected = dv * np.sqrt(2.0**2 + 3.0**2)
    np.testing.assert_allclose(mom0, expected)


def test_dirty_noise_cube_rms_matches_mc_scale():
    """Smoke-test MC dirty noise without requiring a real NUFFT transformer."""

    class _FakeTransformer:
        def image_from(self, visibilities):
            # Map complex visibility mean amplitude into a constant image.
            arr = np.asarray(visibilities.array)
            val = float(np.mean(np.abs(arr)))
            return np.full((3, 3), val, dtype=float)

    from src.utils import autolens_utils

    # noise_map shape: (n_chan, n_vis, 2) like LensKin visibility σ
    sigma = np.ones((2, 5, 2), dtype=float) * 0.5
    transformers = [_FakeTransformer(), _FakeTransformer()]
    rms = autolens_utils.dirty_noise_cube_from(
        noise_map=sigma,
        transformers=transformers,
        n_realizations=20,
        seed=1,
    )
    assert rms.shape == (2, 3, 3)
    assert np.all(rms > 0.0)
    # Larger visibility σ → larger dirty rms
    sigma2 = sigma * 3.0
    rms2 = autolens_utils.dirty_noise_cube_from(
        noise_map=sigma2,
        transformers=transformers,
        n_realizations=20,
        seed=1,
    )
    np.testing.assert_allclose(rms2 / rms, 3.0, rtol=0.2)
