"""
Build phase-1 datasets from a multi-channel visibility cube.

Autolens interferometer fits treat real and imaginary visibility components as
independent Gaussian terms with standard deviations ``sigma_re`` and ``sigma_im``.

Two collapse strategies are supported via ``collapse_mode`` in settings:

- ``concatenate``: stack every channel's visibilities and UV coordinates into one
  dataset (each visibility keeps its native UV and per-channel sigma). Preferred
  for kinematic cubes.
- ``mean``: complex mean over channels with propagated mean noise and averaged UV.
"""

import numpy as np


def concatenate_visibilities_to_moment0(visibilities, sigma, *, sigma_scale=1.0):
    """
    Stack all channels into one visibility vector, preserving per-channel sigma.

    Returns complex visibilities shape ``(n_channels * n_vis,)`` and sigma shape
    ``(n_channels * n_vis, 2)``.
    """
    visibilities = np.asarray(visibilities)
    sigma = np.asarray(sigma)
    vis_complex = visibilities[..., 0] + 1j * visibilities[..., 1]
    vis_flat = vis_complex.reshape(-1)
    sigma_flat = sigma.reshape(-1, 2) * sigma_scale
    return vis_flat, sigma_flat


def concatenate_uv_to_moment0(uv_wavelengths):
    """Stack per-channel UV coordinates into one ``(n_channels * n_vis, 2)`` array."""
    uv_wavelengths = np.asarray(uv_wavelengths)
    return uv_wavelengths.reshape(-1, uv_wavelengths.shape[-1])


def moment0_arrays_from(
    uv_wavelengths,
    visibilities,
    sigma,
    *,
    moment0_settings=None,
    weights=None,
):
    """
    Return ``(visibilities, sigma, uv_wavelengths)`` for a phase-1 dataset.

    Dispatches on ``collapse_mode`` in ``moment0_settings`` (default ``mean``).
    """
    moment0_settings = moment0_settings or {}
    collapse_mode = moment0_settings.get("collapse_mode", "mean")
    sigma_scale = float(moment0_settings.get("sigma_scale", 1.0))

    if collapse_mode == "concatenate":
        vis_out, sigma_out = concatenate_visibilities_to_moment0(
            visibilities=visibilities,
            sigma=sigma,
            sigma_scale=sigma_scale,
        )
        uv_out = concatenate_uv_to_moment0(uv_wavelengths)
        return vis_out, sigma_out, uv_out

    if collapse_mode == "mean":
        vis_out, sigma_out = collapse_visibilities_to_moment0(
            visibilities=visibilities,
            sigma=sigma,
            sigma_mode=moment0_settings.get("sigma_mode", "independent_mean"),
            sigma_scale=sigma_scale,
            weights=weights,
        )
        uv_out = collapse_uv_to_moment0(
            uv_wavelengths=uv_wavelengths,
            uv_mode=moment0_settings.get("uv_mode", "average"),
        )
        return vis_out, sigma_out, uv_out

    raise ValueError(
        f"Unsupported collapse_mode: {collapse_mode!r}. "
        "Choose 'concatenate' or 'mean'."
    )


def collapse_visibilities_to_moment0(
    visibilities,
    sigma,
    *,
    sigma_mode="independent_mean",
    sigma_scale=1.0,
    weights=None,
):
    """
    Collapse channel axis to a velocity-averaged visibility vector and noise map.

    Parameters
    ----------
    visibilities
        Shape ``(n_channels, n_vis, 2)`` with real/imag on the last axis.
    sigma
        Same shape as ``visibilities``.
    sigma_mode
        How to propagate per-channel uncertainties to the channel mean:

        - ``independent_mean`` (default): for a visibility mean
          ``v_mean = sum(v_i) / N`` with independent per-channel errors
          ``sigma_i``, use ``sqrt(sum(sigma_i^2)) / N`` on each of real and imag.
        - ``single_channel``: use the central channel ``sigma_ref / sqrt(N)``.
        - ``weights``: derive sigma from CASA weights (``1 / sqrt(weight)``) and
          use ``sqrt(sum(sigma_i^2)) / N``. Requires ``weights``.
    sigma_scale
        Multiplicative calibration factor applied after propagation.
    weights
        Optional CASA-style weights with shape ``(n_channels, n_vis)`` or
        ``(n_channels, n_vis, 2)``.
    """
    visibilities = np.asarray(visibilities)
    sigma = np.asarray(sigma)

    vis_complex = visibilities[..., 0] + 1j * visibilities[..., 1]
    n_channels = vis_complex.shape[0]
    vis_mom0 = vis_complex.mean(axis=0)

    if sigma_mode == "weights":
        if weights is None:
            raise ValueError("sigma_mode='weights' requires a weights array.")
        weights = np.asarray(weights)
        if weights.ndim == 3:
            weight_arr = weights[..., 0]
        else:
            weight_arr = weights
        sigma_re = 1.0 / np.sqrt(weight_arr)
        sigma_im = sigma_re
    else:
        sigma_re = sigma[..., 0]
        sigma_im = sigma[..., 1]

    if sigma_mode in {"independent_mean", "weights"}:
        sigma_mom0_re = np.sqrt(np.sum(sigma_re**2, axis=0)) / n_channels
        sigma_mom0_im = np.sqrt(np.sum(sigma_im**2, axis=0)) / n_channels
    elif sigma_mode == "single_channel":
        ref = n_channels // 2
        sigma_mom0_re = sigma_re[ref] / np.sqrt(n_channels)
        sigma_mom0_im = sigma_im[ref] / np.sqrt(n_channels)
    else:
        raise ValueError(
            f"Unsupported sigma_mode: {sigma_mode!r}. "
            "Choose 'independent_mean', 'single_channel', or 'weights'."
        )

    sigma_mom0 = np.stack(
        (sigma_mom0_re * sigma_scale, sigma_mom0_im * sigma_scale),
        axis=-1,
    )
    return vis_mom0, sigma_mom0


def collapse_uv_to_moment0(uv_wavelengths, uv_mode="average"):
    """
    Collapse per-channel UV coordinates for a moment-0 dataset.

    ``average`` is preferred for narrow-band data; ``reference_channel`` keeps
    the legacy behaviour of using only the central channel.
    """
    uv_wavelengths = np.asarray(uv_wavelengths)
    if uv_mode == "reference_channel":
        ref = uv_wavelengths.shape[0] // 2
        return uv_wavelengths[ref]
    if uv_mode == "average":
        return uv_wavelengths.mean(axis=0)
    raise ValueError(
        f"Unsupported uv_mode: {uv_mode!r}. Choose 'average' or 'reference_channel'."
    )


def moment0_settings_from_reconstruction(rec_cfg):
    moment0_cfg = rec_cfg.get("moment0", {})
    return {
        "collapse_mode": moment0_cfg.get("collapse_mode", "mean"),
        "sigma_mode": moment0_cfg.get("sigma_mode", "independent_mean"),
        "sigma_scale": float(moment0_cfg.get("sigma_scale", 1.0)),
        "uv_mode": moment0_cfg.get("uv_mode", "average"),
    }


def channel_scatter_around_mean(visibilities):
    """Empirical per-visibility scatter of channels around their mean (diagnostic)."""
    visibilities = np.asarray(visibilities)
    vis_complex = visibilities[..., 0] + 1j * visibilities[..., 1]
    vis_mean = vis_complex.mean(axis=0, keepdims=True)
    return np.std(vis_complex - vis_mean, axis=0)
