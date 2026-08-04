"""
Gaussian primary-beam attenuation for ALMA-like interferometric forward models.

The ALMA primary beam is well approximated by a Gaussian with half-power
beam width (HPBW) of 1.13 * λ / D, where D = 12 m for the 12-m array.

This module provides:

  * ``primary_beam_image`` — build a 2-D Gaussian PB map on a real-space grid.
  * ``primary_beam_from_settings`` — convenience wrapper driven by the JSON
    settings ``primary_beam`` block.

The PB is applied as a diagonal image-plane operator (element-wise multiply)
before the NUFFT, following Stacey et al. (2024), §3.2.

References
----------
ALMA Cycle 7 Technical Handbook (§10.5.2)
Stacey et al. 2024, A&A, arXiv:2403.04850
"""
from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT_M_S = 299792458.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def hpbw_arcsec(frequency_hz: float, dish_diameter_m: float = 12.0) -> float:
    """HPBW = 1.13 * λ/D converted to arcseconds."""
    wavelength_m = SPEED_OF_LIGHT_M_S / frequency_hz
    hpbw_rad = 1.13 * wavelength_m / dish_diameter_m
    return np.degrees(hpbw_rad) * 3600.0


def primary_beam_image(
    grid_arcsec: np.ndarray,
    frequency_hz: float,
    dish_diameter_m: float = 12.0,
    pointing_arcsec: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Gaussian primary-beam map on a 2-D real-space grid.

    Parameters
    ----------
    grid_arcsec : (N, N, 2) or (N*N, 2) array
        Pixel coordinates in arcseconds. Last axis is (y, x) following the
        Autolens / autoarray convention.
    frequency_hz : float
        Central observing frequency in Hz.
    dish_diameter_m : float
        Antenna diameter (default 12 m for ALMA).
    pointing_arcsec : (y, x)
        Pointing centre offset in arcseconds (default phase centre = origin).

    Returns
    -------
    pb : ndarray, same spatial shape as *grid_arcsec* minus the last axis.
        Values in [0, 1]; unity at the pointing centre.
    """
    fwhm = hpbw_arcsec(frequency_hz, dish_diameter_m)
    sigma = fwhm * FWHM_TO_SIGMA

    dy = grid_arcsec[..., 0] - pointing_arcsec[0]
    dx = grid_arcsec[..., 1] - pointing_arcsec[1]
    r2 = dy ** 2 + dx ** 2

    return np.exp(-0.5 * r2 / sigma ** 2)


def primary_beam_from_settings(
    settings: dict,
    frequencies_hz: np.ndarray,
    mask_2d,
) -> np.ndarray | None:
    """
    Build a 2-D PB image from the ``primary_beam`` settings block.

    Parameters
    ----------
    settings : dict
        Runner settings; looks for ``settings["primary_beam"]``.
    frequencies_hz : ndarray
        Channel frequencies in Hz (only the mean is used).
    mask_2d : autoarray Mask2D
        Image-plane mask whose ``.derive_grid.all_false`` provides pixel
        coordinates in arcseconds.

    Returns
    -------
    pb : ndarray (N, N) or None
        ``None`` when ``primary_beam.enabled`` is false or absent.
    """
    pb_cfg = settings.get("primary_beam", {})
    if not pb_cfg.get("enabled", False):
        return None

    dish_diameter_m = pb_cfg.get("dish_diameter_m", 12.0)
    pointing = tuple(pb_cfg.get("pointing_arcsec", [0.0, 0.0]))

    centre_freq_hz = float(np.mean(frequencies_hz))

    grid = mask_2d.derive_grid.all_false
    grid_native = np.asarray(grid.native)

    pb = primary_beam_image(
        grid_arcsec=grid_native,
        frequency_hz=centre_freq_hz,
        dish_diameter_m=dish_diameter_m,
        pointing_arcsec=pointing,
    )
    return pb
