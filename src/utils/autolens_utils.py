import os, sys
import numpy as np

# NOTE:
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# NOTE:
# from src.grid.grid import (
#     Grid3D,
# )
from src.grid.grid import Grid3D
from src.mask.mask import (
    Mask3D,
)

# Radians → arcsec (IAU exact: 180/π * 3600).
_RADIANS_TO_ARCSEC = 180.0 / np.pi * 3600.0


def max_uv_distance_wavelengths(uv_wavelengths):
    """
    Longest projected baseline in the UV array, in wavelengths.

    ``uv_wavelengths`` has shape ``(..., 2)`` with ``(u, v)`` in units of
    baseline / λ (as exported by dataprep).
    """
    uv = np.asarray(uv_wavelengths, dtype=float)
    if uv.ndim < 1 or uv.shape[-1] != 2:
        raise ValueError(
            f"Expected uv_wavelengths with last axis length 2; got shape {uv.shape}"
        )
    return float(np.nanmax(np.hypot(uv[..., 0], uv[..., 1])))


def nyquist_pixel_scale_arcsec_from_uv(uv_wavelengths):
    """
    Image-plane pixel scale that Nyquist-samples the longest baseline.

    ``Δθ = 0.5 * λ / b_max`` (radians), with ``b_max / λ = max√(u²+v²)``.
    Returned in arcsec.
    """
    uv_max = max_uv_distance_wavelengths(uv_wavelengths)
    if not np.isfinite(uv_max) or uv_max <= 0.0:
        raise ValueError(
            f"Cannot derive Nyquist pixel scale: max UV distance is {uv_max}"
        )
    return (0.5 / uv_max) * _RADIANS_TO_ARCSEC


def source_grid_bounding_box_from_cfg(source_cfg):
    """
    Return ``[y_min, y_max, x_min, x_max]`` in arcsec, or ``None``.

    Accepts either a ``bounding_box`` list or explicit ``xmin``/``xmax``/
    ``ymin``/``ymax`` keys.
    """
    if "bounding_box" in source_cfg:
        return list(source_cfg["bounding_box"])
    keys = ("ymin", "ymax", "xmin", "xmax")
    if all(key in source_cfg for key in keys):
        return [
            source_cfg["ymin"],
            source_cfg["ymax"],
            source_cfg["xmin"],
            source_cfg["xmax"],
        ]
    return None


def source_grid_from_settings(settings):
    """
    KinMS source-plane grid (phase-2 cube generation).

    Defaults to ``DEFAULT_KINMS_SOURCE_N_PIXELS`` (256) and top-level
    ``real_space_width``. The image-plane ``n_pixels`` is **not** reused here —
    a coarse fit grid (e.g. 40²) must not drive KinMS resolution. An optional
    ``source_grid`` block can override ``n_pixels``, ``real_space_width``, or
    define a bounding box without changing the image-plane grid.
    """
    source_cfg = settings.get("source_grid", {})
    n_pixels = source_cfg.get("n_pixels", kinms_source_n_pixels_from_settings(settings))
    real_space_width = source_cfg.get("real_space_width", settings["real_space_width"])
    pixel_scale = source_cfg.get(
        "pixel_scale",
        settings.get("pixel_scale", real_space_width / n_pixels),
    )
    return n_pixels, pixel_scale, real_space_width


DEFAULT_KINMS_SOURCE_N_PIXELS = 256


def kinms_source_n_pixels_from_settings(settings):
    """Per-axis resolution of the phase-2 KinMS source grid."""
    return int(
        settings.get("source_grid", {}).get(
            "n_pixels", DEFAULT_KINMS_SOURCE_N_PIXELS
        )
    )


def kinms_source_grid_3d_from_settings(settings, n_channels: int) -> Grid3D:
    """
    Build the KinMS source-plane ``Grid3D``.

    If ``source_grid`` defines a bounding box (``bounding_box`` or
    ``xmin``/``xmax``/``ymin``/``ymax``), use :meth:`Grid3D.bounding_box`.
    Otherwise fall back to a uniform grid from :func:`source_grid_from_settings`.
    """
    source_cfg = settings.get("source_grid", {})
    bounding_box = source_grid_bounding_box_from_cfg(source_cfg)
    if bounding_box is not None:
        n_pixels = source_cfg.get("n_pixels")
        if n_pixels is None:
            raise ValueError(
                "source_grid bounding box requires 'n_pixels' (e.g. 512)."
            )
        return Grid3D.bounding_box(
            bounding_box=bounding_box,
            n_pixels=int(n_pixels),
            n_channels=n_channels,
        )

    n_pixels, pixel_scale, _ = source_grid_from_settings(settings)
    return Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=n_channels,
    )


def kinms_source_grid_3d(settings, n_channels: int, phase1_result=None) -> Grid3D:
    """
    KinMS source-plane grid for phase 2.

    **Parametric** (``phase1_result is None``): built from ``settings['source_grid']``
    — either a uniform field (``n_pixels`` + ``real_space_width``) or an explicit
    bounding box (``xmin``/``xmax``/``ymin``/``ymax`` + ``n_pixels``).

    **After phase 1** (``phase1_result`` set): the bounding box comes from the
    Autolens reconstruction mesh; only ``source_grid.n_pixels`` from settings sets
    the KinMS grid resolution.
    """
    if phase1_result is not None:
        from src.pipelines.reconstruction import phase2_source_grid_from_result

        return phase2_source_grid_from_result(
            result=phase1_result,
            n_channels=n_channels,
            settings=settings,
        )
    return kinms_source_grid_3d_from_settings(settings, n_channels=n_channels)


def source_grid_label_from_settings(settings, phase1_result=None):
    """Short human-readable description of the KinMS source grid."""
    if phase1_result is not None:
        from src.pipelines.reconstruction import source_plane_bounding_box_from_result

        y_min, y_max, x_min, x_max = source_plane_bounding_box_from_result(
            phase1_result
        )
        n_pixels = kinms_source_n_pixels_from_settings(settings)
        return (
            f"{n_pixels}² phase-1 bbox "
            f"x=[{x_min:.3f}, {x_max:.3f}]″ y=[{y_min:.3f}, {y_max:.3f}]″"
        )

    source_cfg = settings.get("source_grid", {})
    bounding_box = source_grid_bounding_box_from_cfg(source_cfg)
    if bounding_box is not None:
        y_min, y_max, x_min, x_max = bounding_box
        n_pixels = source_cfg.get("n_pixels", "?")
        return (
            f"{n_pixels}² bbox "
            f"x=[{x_min}, {x_max}]″ y=[{y_min}, {y_max}]″"
        )
    n_pixels, pixel_scale, width = source_grid_from_settings(settings)
    return f"{n_pixels}² ({width}″ field, {pixel_scale:.4g}″/pix)"


def image_plane_grid_from_settings(settings, uv_wavelengths=None):
    """
    Image-plane grid for transformers, lensing evaluation, and dirty images.

    Always uses top-level ``n_pixels`` (not ``source_grid``). Pixel scale:

    - Numeric ``pixel_scale``: use that value (arcsec).
    - ``pixel_scale: "nyquist"`` / ``"auto"``, or omitted when UV is provided
      (default): Nyquist sampling ``0.5 * λ/b_max`` from the longest baseline.
      Field of view is then ``n_pixels * pixel_scale`` (keeps the coarse
      ``n_pixels`` grid, e.g. 40²).
    - Omitted with no UV: ``real_space_width / n_pixels`` (legacy).

    Override the Nyquist default with ``"pixel_scale_mode": "fov"`` to force
    ``real_space_width / n_pixels`` even when UV is available.
    """
    n_pixels = int(settings["n_pixels"])
    explicit = settings.get("pixel_scale", None)
    width = settings.get("real_space_width", None)
    mode = settings.get("pixel_scale_mode", None)

    if isinstance(explicit, (int, float)):
        pixel_scale = float(explicit)
        if width is None:
            width = n_pixels * pixel_scale
        else:
            width = float(width)
        return n_pixels, pixel_scale, width

    want_nyquist = False
    if isinstance(explicit, str) and explicit.lower() in {"nyquist", "auto"}:
        want_nyquist = True
    elif mode is not None:
        want_nyquist = str(mode).lower() in {"nyquist", "auto"}
    elif explicit is None and uv_wavelengths is not None:
        # Default: Nyquist from UV when baselines are available.
        want_nyquist = str(mode or "nyquist").lower() in {"nyquist", "auto"}

    if mode is not None and str(mode).lower() in {"fov", "width", "real_space_width"}:
        want_nyquist = False

    if want_nyquist:
        if uv_wavelengths is None:
            raise ValueError(
                "Nyquist image-plane pixel_scale requires uv_wavelengths "
                "(or set a numeric pixel_scale / pixel_scale_mode='fov')."
            )
        pixel_scale = nyquist_pixel_scale_arcsec_from_uv(uv_wavelengths)
        # Keep n_pixels fixed (coarse DFT/NUFFT grid); FOV follows Nyquist.
        width = n_pixels * pixel_scale
        return n_pixels, pixel_scale, width

    if width is None:
        raise ValueError(
            "settings must provide real_space_width, a numeric pixel_scale, "
            "or uv_wavelengths for Nyquist sampling."
        )
    width = float(width)
    pixel_scale = width / n_pixels
    return n_pixels, pixel_scale, width


def resolve_image_plane_grid_in_settings(settings, uv_wavelengths):
    """
    Resolve Nyquist / FOV image-plane scale into concrete numeric settings.

    Mutates ``settings`` in place so mask construction, plots, and KinMS FOV
    helpers that read ``real_space_width`` / ``pixel_scale`` stay consistent.
    Returns ``(n_pixels, pixel_scale, real_space_width)``.
    """
    n_pixels, pixel_scale, width = image_plane_grid_from_settings(
        settings, uv_wavelengths=uv_wavelengths
    )
    settings["n_pixels"] = int(n_pixels)
    settings["pixel_scale"] = float(pixel_scale)
    settings["real_space_width"] = float(width)
    return n_pixels, pixel_scale, width


def image_plane_mask_from_settings(settings, uv_wavelengths=None):
    n_pixels, pixel_scale, _ = image_plane_grid_from_settings(
        settings, uv_wavelengths=uv_wavelengths
    )
    return al.Mask2D.all_false(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=pixel_scale,
    )


def image_extent_arcsec(real_space_width):
    half_width = real_space_width / 2.0
    return (-half_width, half_width, -half_width, half_width)


def image_extent_from_bounding_box(bounding_box):
    y_min, y_max, x_min, x_max = bounding_box
    return (x_min, x_max, y_min, y_max)


def nufftax_is_usable():
    """Return True if JAX-native nufftax NUFFT can be imported."""
    try:
        from src.utils.jax_compat import jax_is_usable, using_numpy_jax_stub

        if using_numpy_jax_stub() or not jax_is_usable():
            return False
    except Exception:
        return False
    try:
        import nufftax  # noqa: F401

        return True
    except Exception:
        return False


def default_transformer_class():
    """
    Prefer JAX-native ``TransformerNUFFT`` (nufftax) when available.

    Fallback order:
      1. ``TransformerNUFFT`` — real jax + nufftax
      2. ``TransformerNUFFTPyNUFFT`` — pynufft (no JAX)
      3. ``TransformerDFT`` — always available (slow for large UV sets)
    """
    if nufftax_is_usable():
        return al.TransformerNUFFT

    try:
        import pynufft  # noqa: F401

        return al.TransformerNUFFTPyNUFFT
    except Exception:
        return al.TransformerDFT


def transformer_class_from_settings(settings=None):
    """
    Resolve transformer class from settings.

    Optional keys (first match wins):
      - top-level ``transformer``
      - ``reconstruction.transformer``

    Values: ``auto`` (default), ``dft``, ``nufft`` / ``nufftax``, ``pynufft``.
    """
    name = None
    if settings is not None:
        name = settings.get("transformer")
        if name is None:
            name = settings.get("reconstruction", {}).get("transformer")
    if name is None or str(name).lower() in {"auto", "default"}:
        return default_transformer_class()

    key = str(name).lower()
    if key == "dft":
        return al.TransformerDFT
    if key in {"nufft", "nufftax"}:
        return al.TransformerNUFFT
    if key in {"pynufft", "nufft_pynufft"}:
        return al.TransformerNUFFTPyNUFFT
    raise ValueError(
        f"Unsupported transformer: {name!r}. "
        "Expected 'auto', 'dft', 'nufft'/'nufftax', or 'pynufft'."
    )


def transformer_class_with_primary_beam(base_class, primary_beam):
    """
    Return a callable with the ``(uv_wavelengths, real_space_mask)`` signature
    expected by ``al.Interferometer``, wrapping each instance in
    :class:`TransformerWithPrimaryBeam`.
    """
    def _factory(uv_wavelengths, real_space_mask, **kwargs):
        base = base_class(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            **kwargs,
        )
        return TransformerWithPrimaryBeam(base, primary_beam)
    _factory.__name__ = f"{base_class.__name__}+PB"
    return _factory


def transformer_from_uv_and_settings(
    uv_wavelengths,
    settings,
    transformer_class=None,
):
    if transformer_class is None:
        transformer_class = transformer_class_from_settings(settings)
    return transformer_class(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=image_plane_mask_from_settings(settings),
    )


def dirty_image_from_visibilities(visibilities, uv_wavelengths, settings):
    """
    Dirty image on the settings image-plane grid (``n_pixels``, ``real_space_width``).
    """
    transformer = transformer_from_uv_and_settings(
        uv_wavelengths=uv_wavelengths,
        settings=settings,
    )
    if not isinstance(visibilities, al.Visibilities):
        visibilities = al.Visibilities(visibilities=visibilities)
    return array2d_to_numpy(
        transformer.image_from(visibilities=visibilities)
    )


class TransformerWithPrimaryBeam:
    """
    Thin wrapper that multiplies the real-space image by a primary-beam map
    before delegating to the underlying transformer.

    The adjoint (``image_from``) is *not* divided by the PB — dirty images
    remain in attenuated (observed) units, consistent with the data.
    """

    def __init__(self, base_transformer, primary_beam):
        self._base = base_transformer
        self._pb = primary_beam

    def __getattr__(self, name):
        return getattr(self._base, name)

    def visibilities_from(self, image, **kwargs):
        pb_image = al.Array2D(
            values=np.asarray(image) * self._pb,
            mask=self._base.real_space_mask,
        )
        return self._base.visibilities_from(image=pb_image, **kwargs)

    def image_from(self, visibilities, **kwargs):
        return self._base.image_from(visibilities=visibilities, **kwargs)


def transformers_from(
    uv_wavelengths,
    mask_3d: Mask3D,
    transformer_class=None,
    settings=None,
    primary_beam=None,
):
    if transformer_class is None:
        transformer_class = (
            transformer_class_from_settings(settings)
            if settings is not None
            else default_transformer_class()
        )

    transformers = []
    for i in range(mask_3d.n_channels):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            real_space_mask=mask_3d.mask_2d
        )
        if primary_beam is not None:
            transformer = TransformerWithPrimaryBeam(transformer, primary_beam)
        transformers.append(transformer)

    return transformers


def visibilities_from_transformers_and_cube(
    cube: np.ndarray,
    transformers: list,
    shape,
    z_mask=None,
    primary_beam=None,
):
    """
    Forward-model an image cube to visibilities via per-channel transformers.

    Parameters
    ----------
    primary_beam : ndarray (N, N), optional
        If supplied, each channel image is multiplied by this map before the
        Fourier transform (Gaussian antenna response / primary beam).
    """
    if z_mask is None:
        pass

    visibilities = np.zeros(shape=shape)
    for i, transformer in enumerate(transformers):
        if transformer is not None:
            channel = cube[i]
            if primary_beam is not None:
                channel = channel * primary_beam
            image = al.Array2D(
                values=channel,
                mask=transformer.real_space_mask,
            )
            visibilities_i = transformer.visibilities_from(image=image)
            vis_array = visibilities_i.array
            visibilities[i, :, 0] = vis_array.real
            visibilities[i, :, 1] = vis_array.imag
        else:
            raise NotImplementedError()

    return visibilities


def array2d_to_numpy(array_2d):
    """Convert an autoarray ``Array2D`` (or native view) to a plain 2D ndarray."""
    if hasattr(array_2d, "native"):
        array_2d = array_2d.native
    return np.asarray(getattr(array_2d, "array", array_2d))


def dirty_cube_from(
    visibilities: np.ndarray,
    transformers: list
):
    """
    Generate an image cube from visibilities.

    Notes
    -----
    `autolens` transformer APIs expect a visibilities object with a `.array`
    attribute (e.g. `al.Visibilities`), not a raw complex `numpy.ndarray`.

    Returned arrays use autolens image orientation and should be plotted with
    ``origin="lower"``.
    """

    return np.array([
        array2d_to_numpy(
            transformer.image_from(
                visibilities=al.Visibilities(
                    visibilities=visibilities[i, :, 0] + 1j * visibilities[i, :, 1]
                )
            )
        )
        for i, transformer in enumerate(transformers)
    ])


def dirty_noise_cube_from(
    noise_map: np.ndarray,
    transformers: list,
    *,
    n_realizations: int = 8,
    seed: int = 0,
):
    """
    Monte-Carlo estimate of the dirty-image noise cube from visibility σ.

    Draws ``n_realizations`` Gaussian noise visibility cubes
    ``N(0, noise_map)``, dirty-images each, and returns the per-pixel RMS
    over realizations (same shape as :func:`dirty_cube_from`).
    """
    sigma = np.asarray(noise_map, dtype=float)
    rng = np.random.RandomState(int(seed))
    acc = None
    n_realizations = max(int(n_realizations), 1)
    for _ in range(n_realizations):
        noise = rng.normal(size=sigma.shape) * sigma
        dirty = np.asarray(dirty_cube_from(noise, transformers), dtype=float)
        if acc is None:
            acc = dirty**2
        else:
            acc += dirty**2
    return np.sqrt(acc / float(n_realizations))


def _visibility_sigma_re_im(noise_map):
    """
    Extract real/imag visibility σ from an Autolens noise map or ndarray.

    ``VisibilitiesNoiseMap`` stores ``σ_re + 1j σ_im``. A float array of shape
    ``(n_vis, 2)`` is also accepted.
    """
    sigma = np.asarray(noise_map)
    if np.iscomplexobj(sigma):
        return np.real(sigma).astype(float), np.imag(sigma).astype(float)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2 and sigma.shape[-1] == 2:
        return sigma[..., 0], sigma[..., 1]
    if sigma.ndim == 1:
        return sigma, sigma
    raise ValueError(
        f"Unsupported visibility noise_map shape {sigma.shape} / dtype {sigma.dtype}"
    )


def dirty_noise_map_mc_from(
    noise_map,
    transformer,
    *,
    n_realizations: int = 8,
    seed: int = 0,
):
    """
    Monte-Carlo RMS dirty image from a single-channel visibility noise map.

    Prefer this over ``FitInterferometer.dirty_noise_map``, which is only the
    inverse Fourier transform of the noise-map values (can be negative / zero)
    and is **not** an image-plane RMS.

    Pixels outside the transformer's real-space mask are left at 0 (no
    coverage); callers should treat non-positive values as invalid when
    forming residual / σ maps.
    """
    sig_re, sig_im = _visibility_sigma_re_im(noise_map)
    rng = np.random.RandomState(int(seed))
    n_realizations = max(int(n_realizations), 1)
    acc = None
    for _ in range(n_realizations):
        noise_vis = (
            rng.normal(size=sig_re.shape) * sig_re
            + 1j * rng.normal(size=sig_im.shape) * sig_im
        )
        dirty = array2d_to_numpy(
            transformer.image_from(
                visibilities=al.Visibilities(visibilities=noise_vis)
            )
        )
        dirty = np.asarray(dirty, dtype=float)
        if acc is None:
            acc = dirty**2
        else:
            acc += dirty**2
    return np.sqrt(acc / float(n_realizations))


def dirty_noise_map_mc_from_fit(fit, *, n_realizations: int = 8, seed: int = 0):
    """Monte-Carlo dirty-image RMS for a phase-1 ``FitInterferometer``."""
    return dirty_noise_map_mc_from(
        noise_map=fit.dataset.noise_map,
        transformer=fit.dataset.transformer,
        n_realizations=n_realizations,
        seed=seed,
    )


def dirty_mom0_noise_from_channel_noise(channel_noise_cube, z_step_kms):
    """
    Propagate per-channel dirty σ to mom0 = Σ_c dirty_c * dv (independent channels).

    ``σ_mom0 = dv * sqrt(Σ_c σ_c²)``.
    """
    sigma = np.asarray(channel_noise_cube, dtype=float)
    dv = float(z_step_kms)
    return dv * np.sqrt(np.sum(sigma**2, axis=0))
