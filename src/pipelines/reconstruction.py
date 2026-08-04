import autofit as af
import autolens as al
import logging
import numpy as np
from astropy import units
from scipy.interpolate import griddata

from src.pipelines.lens_model import (
    apply_lensing_off_reconstruction_overrides,
    lens_galaxy_model_from_settings,
    lensing_enabled,
)
from src.pipelines.moment0_noise import (
    moment0_arrays_from,
    moment0_settings_from_reconstruction,
)
from src.pipelines.search import build_optimizer_from_settings
from src.grid.grid import Grid3D
from src.utils import autolens_utils, spectral_utils

logger = logging.getLogger(__name__)


def _never_visualize(paths, during_analysis=True):
    """Module-level (picklable) replacement for ``Analysis.should_visualize``."""
    return False


def z_step_kms_from_settings(frequencies, settings):
    return spectral_utils.z_step_kms_from_data_frequencies(frequencies)


def moment0_dataset_from(
    uv_wavelengths,
    visibilities,
    sigma,
    mask_2d,
    transformer_class=None,
    moment0_settings=None,
    weights=None,
):
    """
    Build a single-channel ``Interferometer`` dataset for phase-1 reconstruction.

    Default ``collapse_mode`` is ``mean`` (complex channel average). Set
    ``collapse_mode: concatenate`` in ``reconstruction.moment0`` to stack every
    channel's visibilities at native UV coordinates (recommended for kinematic
    cubes).

    The fitted image is in native per-channel units (Jy/pixel). Velocity
    integration (``n_channels * z_step_kms``) is applied later for KinMS.
    """
    if transformer_class is None:
        transformer_class = autolens_utils.default_transformer_class()

    moment0_settings = moment0_settings or {}
    vis_mom0, sigma_mom0, uv_mom0 = moment0_arrays_from(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        moment0_settings=moment0_settings,
        weights=weights,
    )

    data = al.Visibilities(visibilities=vis_mom0)
    noise_map = al.VisibilitiesNoiseMap(visibilities=sigma_mom0)

    # DFT is intentionally used when JAX/nufftax are unavailable; allow large
    # visibility counts (moment-0 / concatenate modes often exceed 10k).
    kwargs = {}
    if transformer_class is al.TransformerDFT:
        kwargs["raise_error_dft_visibilities_limit"] = False

    return al.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_mom0,
        real_space_mask=mask_2d,
        transformer_class=transformer_class,
        **kwargs,
    )


def _build_lens_model(settings):
    rec_cfg = settings["reconstruction"]
    free_centre = False if not lensing_enabled(settings) else not rec_cfg.get(
        "fix_lens", False
    )
    return lens_galaxy_model_from_settings(
        settings,
        free_centre=free_centre,
        centre_prior_cfg=rec_cfg.get("centre_prior"),
    )


_REGULARIZATION_CLASSES = {
    "constant": al.reg.Constant,
    "constant_split": al.reg.ConstantSplit,
    "adapt": al.reg.Adapt,
    "adapt_split": al.reg.AdaptSplit,
    "adapt_split_zeroth": al.reg.AdaptSplitZeroth,
}

_ADAPT_REGULARIZATION_TYPES = frozenset({"adapt", "adapt_split", "adapt_split_zeroth"})

_ADAPT_ZEROTH_REGULARIZATION_TYPES = frozenset({"adapt_split_zeroth"})

_DELAUNAY_ONLY_REGULARIZATION_TYPES = frozenset(
    {"constant_split", "adapt_split", "adapt_split_zeroth"}
)

_DEFAULT_REGULARIZATION_TYPE_BY_MESH = {
    "rectangular_adapt_density": "constant",
    "rectangular_uniform": "constant",
    "rectangular_adapt_image": "constant",
    # Prefer brightness-weighted AdaptSplit on Delaunay: less prone to rogue
    # edge pixels than ConstantSplit at likelihood-optimal λ.
    "delaunay": "adapt_split",
}


def regularization_type_from_settings(rec_cfg):
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")
    reg_cfg = rec_cfg.get("regularization", {})
    return reg_cfg.get(
        "type",
        _DEFAULT_REGULARIZATION_TYPE_BY_MESH.get(mesh_type, "constant"),
    )


def regularization_needs_adapt_image(reg_type):
    return reg_type in _ADAPT_REGULARIZATION_TYPES


def use_positive_only_solver_from_settings(settings, default=False):
    """
    Whether phase-1 inversions use Autolens' positive-only linear solver.

    Read from ``reconstruction.use_positive_only_solver``. Default ``False``
    matches historical LensKin behaviour (source pixels may go negative).
    Regularization type does not imply positivity — this flag does.
    """
    rec_cfg = settings.get("reconstruction", {}) if settings is not None else {}
    return bool(rec_cfg.get("use_positive_only_solver", default))


def validate_mesh_regularization_pair(mesh_type, reg_type):
    """
    ``ConstantSplit`` / ``AdaptSplit`` / ``AdaptSplitZeroth`` use cross-derivative
    regularization that requires a Delaunay interpolator
    (``_mappings_sizes_weights_split``).
    """
    if reg_type in _DELAUNAY_ONLY_REGULARIZATION_TYPES and mesh_type != "delaunay":
        raise ValueError(
            f"reconstruction.regularization.type={reg_type!r} requires "
            f"mesh_type='delaunay' (got {mesh_type!r}). For rectangular meshes "
            "use regularization.type='adapt', or set mesh_type to 'delaunay'."
        )


def _scalar_prior_from_settings(reg_cfg, param_name, defaults):
    """
    Return a fixed float or autofit prior for one regularization parameter.

    Per-parameter bounds may be given as ``reg_cfg[param_name]`` dict with
    ``lower_limit`` / ``upper_limit``, or fall back to ``reg_cfg``-level keys
    like ``{param_name}_lower_limit``.
    """
    prior_type = reg_cfg.get("prior_type", "log_uniform")
    param_cfg = reg_cfg.get(param_name)
    if isinstance(param_cfg, dict):
        if param_cfg.get("prior_type") == "fixed" or "value" in param_cfg:
            return float(param_cfg.get("value", defaults["fixed"]))
        lower_limit = param_cfg.get("lower_limit", defaults["lower"])
        upper_limit = param_cfg.get("upper_limit", defaults["upper"])
        return af.LogUniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)

    if prior_type == "fixed":
        return float(reg_cfg.get(param_name, defaults["fixed"]))

    if prior_type == "gaussian":
        return af.GaussianPrior(
            mean=reg_cfg.get(param_name, reg_cfg.get("initial_value", defaults["fixed"])),
            sigma=reg_cfg.get("sigma", 0.5),
        )

    lower_limit = reg_cfg.get(
        f"{param_name}_lower_limit",
        reg_cfg.get("lower_limit", defaults["lower"]),
    )
    upper_limit = reg_cfg.get(
        f"{param_name}_upper_limit",
        reg_cfg.get("upper_limit", defaults["upper"]),
    )
    return af.LogUniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)


def _regularization_coefficient_from_settings(reg_cfg):
    """
    Return a fixed coefficient or autofit prior for ``Constant`` / ``ConstantSplit``.

    With ``fix_lens: true`` the lens mass is fixed, so regularization is often
    the only free parameter. A wide log-uniform prior (e.g. ``1e-3`` to ``1e3``)
    lets LBFGS drive the coefficient to the lower limit and over-fit the source.

    For production phase-1 on narrow-band data, ``prior_type: fixed`` at
    ``1e5`` is a good starting point. If optimizing, use ``use_jax_gradient:
    true`` in the search block so scipy does not rely on a single finite-
    difference ``eps`` for mixed arcsec + coefficient scales.

    ``prior_type`` options:

    - ``fixed``: use ``value`` (no optimization).
    - ``log_uniform`` (default): bounded log-uniform prior; ball init at unit
      0.5 gives the geometric mean of ``lower_limit`` and ``upper_limit``.
      Defaults ``1e5``–``1e7`` (start ≈ ``1e6``, floor above ``1e4``).
    - ``gaussian``: unbounded Gaussian around ``initial_value``.
    """
    return _scalar_prior_from_settings(
        reg_cfg,
        param_name="coefficient",
        defaults={"fixed": reg_cfg.get("value", 1e6), "lower": 1e5, "upper": 1e7},
    )


def _signal_scale_prior_from_settings(reg_cfg, param_name="signal_scale", default=3.0):
    prior_type = reg_cfg.get("prior_type", "log_uniform")
    param_cfg = reg_cfg.get(param_name)
    if isinstance(param_cfg, dict):
        if param_cfg.get("prior_type") == "fixed" or "value" in param_cfg:
            return float(param_cfg.get("value", default))
        return af.UniformPrior(
            lower_limit=param_cfg.get("lower_limit", 0.0),
            upper_limit=param_cfg.get("upper_limit", 10.0),
        )
    if prior_type == "fixed":
        return float(reg_cfg.get(param_name, default))
    return af.UniformPrior(
        lower_limit=reg_cfg.get(f"{param_name}_lower_limit", 0.0),
        upper_limit=reg_cfg.get(f"{param_name}_upper_limit", 10.0),
    )


def _adapt_regularization_priors_from_settings(reg_cfg, reg_type="adapt"):
    """Priors for ``Adapt`` / ``AdaptSplit`` / ``AdaptSplitZeroth``."""
    priors = {
        "inner_coefficient": _scalar_prior_from_settings(
            reg_cfg,
            param_name="inner_coefficient",
            defaults={"fixed": 0.005, "lower": 1e-6, "upper": 1e6},
        ),
        "outer_coefficient": _scalar_prior_from_settings(
            reg_cfg,
            param_name="outer_coefficient",
            defaults={"fixed": 1.9, "lower": 1e-6, "upper": 1e6},
        ),
        "signal_scale": _signal_scale_prior_from_settings(reg_cfg),
    }
    if reg_type in _ADAPT_ZEROTH_REGULARIZATION_TYPES:
        priors["zeroth_coefficient"] = _scalar_prior_from_settings(
            reg_cfg,
            param_name="zeroth_coefficient",
            defaults={"fixed": 1.0, "lower": 1e-6, "upper": 1e6},
        )
        priors["zeroth_signal_scale"] = _signal_scale_prior_from_settings(
            reg_cfg, param_name="zeroth_signal_scale", default=1.0
        )
    return priors


def _regularization_scalar_from_config(reg_cfg, param_name, default):
    """Extract a concrete float from a scalar or prior dict in settings."""
    param = reg_cfg.get(param_name, default)
    if isinstance(param, dict):
        if "value" in param:
            return float(param["value"])
        lower_limit = param.get("lower_limit")
        upper_limit = param.get("upper_limit")
        if lower_limit is not None and upper_limit is not None:
            return float(np.sqrt(lower_limit * upper_limit))
        return float(default)
    return float(param)


def fixed_regularization_values_from_settings(reg_cfg, reg_type, overrides=None):
    """Return concrete floats for workspace-style fixed fits."""
    overrides = dict(overrides or {})
    if reg_type in _ADAPT_REGULARIZATION_TYPES:
        values = {
            "inner_coefficient": _regularization_scalar_from_config(
                reg_cfg, "inner_coefficient", 0.005
            ),
            "outer_coefficient": _regularization_scalar_from_config(
                reg_cfg, "outer_coefficient", 1.9
            ),
            "signal_scale": _regularization_scalar_from_config(
                reg_cfg, "signal_scale", 3.0
            ),
        }
        if reg_type in _ADAPT_ZEROTH_REGULARIZATION_TYPES:
            values["zeroth_coefficient"] = _regularization_scalar_from_config(
                reg_cfg, "zeroth_coefficient", 1.0
            )
            values["zeroth_signal_scale"] = _regularization_scalar_from_config(
                reg_cfg, "zeroth_signal_scale", 1.0
            )
    else:
        values = {
            "coefficient": _regularization_scalar_from_config(
                reg_cfg, "coefficient", reg_cfg.get("value", 1e6)
            )
        }
        if "value" in reg_cfg and "coefficient" not in reg_cfg:
            values["coefficient"] = float(reg_cfg["value"])
    values.update(overrides)
    return values


def format_regularization_config(reg_cfg, reg_type):
    """Human-readable summary of fixed values or prior ranges from settings."""
    prior_type = reg_cfg.get("prior_type", "log_uniform")
    if prior_type == "fixed" and reg_type not in _ADAPT_REGULARIZATION_TYPES:
        return format_regularization_values(
            reg_type,
            fixed_regularization_values_from_settings(reg_cfg, reg_type),
        )

    if reg_type in _ADAPT_REGULARIZATION_TYPES:
        param_defaults = [
            ("inner_coefficient", 0.005),
            ("outer_coefficient", 1.9),
            ("signal_scale", 3.0),
        ]
        if reg_type in _ADAPT_ZEROTH_REGULARIZATION_TYPES:
            param_defaults.extend(
                [
                    ("zeroth_coefficient", 1.0),
                    ("zeroth_signal_scale", 1.0),
                ]
            )
        parts = []
        for param_name, default in param_defaults:
            param = reg_cfg.get(param_name, default)
            if isinstance(param, dict):
                if param.get("prior_type") == "fixed" or "value" in param:
                    parts.append(
                        f"{param_name}={_regularization_scalar_from_config(reg_cfg, param_name, default)} (fixed)"
                    )
                elif "lower_limit" in param and "upper_limit" in param:
                    lo = param["lower_limit"]
                    hi = param["upper_limit"]
                    init = np.sqrt(lo * hi)
                    parts.append(
                        f"{param_name}: log_uniform [{lo:g}, {hi:g}] (init≈{init:g})"
                    )
                else:
                    parts.append(f"{param_name}={param}")
            else:
                parts.append(f"{param_name}={param}")
        return ", ".join(parts)

    if prior_type == "fixed":
        value = reg_cfg.get("value", reg_cfg.get("coefficient", 1e6))
        return f"coefficient={value} (fixed)"

    lower = reg_cfg.get("lower_limit", 1e5)
    upper = reg_cfg.get("upper_limit", 1e7)
    init = np.sqrt(lower * upper)
    return f"coefficient: log_uniform [{lower:g}, {upper:g}] (init≈{init:g})"


def format_regularization_values(reg_type, values):
    """Human-readable summary of regularization parameters in use."""
    if reg_type in _ADAPT_REGULARIZATION_TYPES:
        text = (
            f"inner_coefficient={values['inner_coefficient']}, "
            f"outer_coefficient={values['outer_coefficient']}, "
            f"signal_scale={values['signal_scale']}"
        )
        if reg_type in _ADAPT_ZEROTH_REGULARIZATION_TYPES:
            text += (
                f", zeroth_coefficient={values['zeroth_coefficient']}, "
                f"zeroth_signal_scale={values['zeroth_signal_scale']}"
            )
        return text
    return f"coefficient={values['coefficient']}"


def _build_regularization_model(reg_cfg, reg_type):
    try:
        reg_cls = _REGULARIZATION_CLASSES[reg_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported regularization type: {reg_type!r}. "
            f"Choose from {sorted(_REGULARIZATION_CLASSES)}."
        ) from exc

    regularization = af.Model(reg_cls)
    if reg_type in _ADAPT_REGULARIZATION_TYPES:
        for param_name, prior in _adapt_regularization_priors_from_settings(
            reg_cfg, reg_type=reg_type
        ).items():
            setattr(regularization, param_name, prior)
    else:
        regularization.coefficient = _regularization_coefficient_from_settings(reg_cfg)
    return regularization


def build_reconstruction_model(settings, mask_2d):
    apply_lensing_off_reconstruction_overrides(settings)
    rec_cfg = settings["reconstruction"]
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")
    reg_cfg = rec_cfg.get("regularization", {})
    reg_type = regularization_type_from_settings(rec_cfg)
    validate_mesh_regularization_pair(mesh_type, reg_type)

    lens = _build_lens_model(settings)

    mesh = _build_mesh_model(settings, mask_2d)
    regularization = _build_regularization_model(reg_cfg, reg_type)
    pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)

    source = af.Model(
        al.Galaxy,
        redshift=settings["redshift_source"],
        pixelization=pixelization,
    )

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


_MESH_MODEL_CLASSES = {
    "rectangular_adapt_density": al.mesh.RectangularAdaptDensity,
    "rectangular_uniform": al.mesh.RectangularUniform,
    "rectangular_adapt_image": al.mesh.RectangularAdaptImage,
}

def reconstruction_mask_from_settings(settings):
    """
    Circular mask for phase-1 Autolens interferometer imaging.

    Field of view follows the **image-plane** ``real_space_width`` (Nyquist-
    resolved when applicable). ``mask_n_pixels`` may oversample that FOV for
    the inversion; it does not change the transformer image-plane grid.
    """
    rec_cfg = settings["reconstruction"]
    _, _, real_space_width = autolens_utils.image_plane_grid_from_settings(settings)
    mask_n_pixels = int(rec_cfg.get("mask_n_pixels", settings["n_pixels"]))
    pixel_scale = real_space_width / mask_n_pixels

    mask_radius = rec_cfg.get("mask_radius", real_space_width / 2.0)

    return al.Mask2D.circular(
        shape_native=(mask_n_pixels, mask_n_pixels),
        pixel_scales=pixel_scale,
        radius=mask_radius,
    )


def _shape_native_scaled_interior_from_grid(grid):
    """
    Same as ``Grid2D.shape_native_scaled_interior``, but via ``.array``.

    Indexing an autoarray structure (``grid[:, 0]``) always imports
    ``jax.numpy``. On hosts where jaxlib was built with AVX and the CPU lacks
    it, that import raises ``RuntimeError``. Using ``.array`` stays on NumPy.
    """
    values = np.asarray(grid.array)
    return (
        float(np.amax(values[:, 0]) - np.amin(values[:, 0])),
        float(np.amax(values[:, 1]) - np.amin(values[:, 1])),
    )


def _overlay_image_plane_mesh_grid(mask_2d, shape):
    """
    Build an Overlay mesh grid without triggering a JAX import.

    Mirrors ``autoarray.inversion.mesh.image_mesh.overlay.Overlay``.
    """
    from autoarray.geometry import geometry_util
    from autoarray.inversion.mesh.image_mesh import overlay as overlay_mod
    from autoarray.structures.grids import grid_2d_util
    from autoarray.structures.grids.irregular_2d import Grid2DIrregular

    shape = (int(shape[0]), int(shape[1]))
    pixel_scales = mask_2d.pixel_scales
    grid = mask_2d.derive_grid.unmasked
    interior = _shape_native_scaled_interior_from_grid(grid)
    overlay_pixel_scales = (
        (interior[0] + pixel_scales[0]) / shape[0],
        (interior[1] + pixel_scales[1]) / shape[1],
    )
    unmasked_overlay_grid = grid_2d_util.grid_2d_slim_via_shape_native_from(
        shape_native=shape,
        pixel_scales=overlay_pixel_scales,
        origin=mask_2d.mask_centre,
    )
    overlaid_centres = np.array(
        geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=unmasked_overlay_grid,
            shape_native=mask_2d.shape_native,
            pixel_scales=mask_2d.pixel_scales,
        )
    ).astype("int")
    mask_array = np.asarray(mask_2d.array)
    total_pixels = overlay_mod.total_pixels_2d_from(
        mask_2d=mask_array,
        overlaid_centres=overlaid_centres,
    )
    overlay_for_mask = overlay_mod.overlay_for_mask_from(
        total_pixels=total_pixels,
        mask=mask_array,
        overlaid_centres=overlaid_centres,
    ).astype("int")
    mesh_grid = overlay_mod.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
    )
    return Grid2DIrregular(values=mesh_grid)


def _image_plane_mesh_grid_from_settings(rec_cfg, mask_2d):
    """
    Build the image-plane vertex grid used by the Delaunay mesh.

    An ``Overlay`` grid is laid over the dataset mask and ray-traced to the
    source plane during the fit. Magnification makes the effective source-plane
    sampling denser in highly magnified regions.
    """
    image_mesh_shape = tuple(
        rec_cfg.get(
            "image_mesh_shape",
            rec_cfg.get("mesh_shape", [28, 28]),
        )
    )
    edge_pixels = int(rec_cfg.get("delaunay_edge_pixels", 30))
    mask_radius = rec_cfg.get("mask_radius")
    if mask_radius is None:
        mask_radius = mask_2d.circular_radius

    # Avoid Overlay.image_plane_mesh_grid_from: it indexes Grid2D and forces a
    # jax.numpy import even when phase-1 use_jax is False.
    image_plane_mesh_grid = _overlay_image_plane_mesh_grid(
        mask_2d, image_mesh_shape
    )
    return al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask_2d.mask_centre,
        radius=mask_radius + mask_2d.pixel_scale / 2.0,
        n_points=edge_pixels,
    )


def adapt_images_for_reconstruction(settings, mask_2d, dataset=None):
    """
    Return ``AdaptImages`` when the mesh or regularization needs external adapt data.

    - ``delaunay``: requires the precomputed image-plane mesh grid.
    - ``rectangular_adapt_image``: dirty-image adapt map for mesh weighting.
    - ``adapt`` / ``adapt_split`` / ``adapt_split_zeroth``: dirty-image adapt map
      for spatially varying regularization (PyAutoLens tutorial 11 / SLaM
      ``source_pix`` stage 2).
    """
    rec_cfg = settings["reconstruction"]
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")
    reg_type = regularization_type_from_settings(rec_cfg)
    source_path = "('galaxies', 'source')"

    image_plane_mesh_grid_dict = {}
    image_dict = {}

    if mesh_type == "delaunay":
        image_plane_mesh_grid_dict[source_path] = _image_plane_mesh_grid_from_settings(
            rec_cfg, mask_2d
        )

    needs_dirty_image = (
        mesh_type == "rectangular_adapt_image"
        or regularization_needs_adapt_image(reg_type)
    )
    if needs_dirty_image:
        if dataset is None:
            raise ValueError(
                f"{mesh_type!r} / {reg_type!r} requires the phase-1 dataset to build "
                "a dirty-image adapt map."
            )
        dirty_image = dataset.dirty_image
        if dirty_image is None:
            dirty_image = dataset.masked_dirty_image
        image_dict[source_path] = dirty_image

    if not image_plane_mesh_grid_dict and not image_dict:
        return None

    kwargs = {}
    if image_dict:
        kwargs["galaxy_name_image_dict"] = image_dict
    if image_plane_mesh_grid_dict:
        kwargs["galaxy_name_image_plane_mesh_grid_dict"] = image_plane_mesh_grid_dict
    return al.AdaptImages(**kwargs)


def _build_mesh_model(settings, mask_2d):
    rec_cfg = settings["reconstruction"]
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")

    if mesh_type == "delaunay":
        grid = _image_plane_mesh_grid_from_settings(rec_cfg, mask_2d)
        edge_pixels = int(rec_cfg.get("delaunay_edge_pixels", 30))
        areas_factor = float(rec_cfg.get("delaunay_areas_factor", 0.5))
        return af.Model(
            al.mesh.Delaunay,
            pixels=grid.shape[0],
            zeroed_pixels=edge_pixels,
            areas_factor=areas_factor,
        )

    try:
        mesh_cls = _MESH_MODEL_CLASSES[mesh_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported reconstruction mesh_type: {mesh_type!r}. "
            f"Choose from {sorted(_MESH_MODEL_CLASSES)} or 'delaunay'."
        ) from exc

    mesh_shape = tuple(rec_cfg.get("mesh_shape", [28, 28]))
    mesh_kwargs = {"shape": mesh_shape}
    if mesh_type == "rectangular_adapt_image":
        if "weight_power" in rec_cfg:
            mesh_kwargs["weight_power"] = rec_cfg["weight_power"]
        if "weight_floor" in rec_cfg:
            mesh_kwargs["weight_floor"] = rec_cfg["weight_floor"]

    return af.Model(mesh_cls, **mesh_kwargs)


class _FixedPhase1Result:
    """Minimal result stand-in when phase 1 has no free parameters."""

    def __init__(self, instance, fit):
        self.max_log_likelihood_instance = instance
        self.max_log_likelihood_fit = fit


def _fmt_centre(centre):
    return f"({float(centre[0]):.6f}, {float(centre[1]):.6f})"


def print_phase1_setup(settings, mask_2d, model, dataset=None, transformer_class=None):
    """Print the phase-1 configuration actually used for the reconstruction."""
    rec_cfg = settings.get("reconstruction", {})
    mass_cfg = settings["lens_mass_model"]
    truth = settings.get("truth_parameters") or {}
    fix_lens = bool(rec_cfg.get("fix_lens", False))
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")
    reg_cfg = rec_cfg.get("regularization", {})
    reg_type = regularization_type_from_settings(rec_cfg)
    img_n, img_pix, img_w = autolens_utils.image_plane_grid_from_settings(settings)
    src_n, src_pix, src_w = autolens_utils.source_grid_from_settings(settings)
    centre_prior = rec_cfg.get("centre_prior")

    print("\n=== Phase-1 parameters (setup) ===")
    print(f"  output_path              = {settings.get('output_path')}")
    print(f"  data_directory           = {settings.get('data_directory')}")
    print(f"  uids / width             = {settings.get('uids')} / {settings.get('width')}")
    print(f"  redshift_lens / source   = {settings.get('redshift_lens')} / {settings.get('redshift_source')}")
    print(
        f"  image-plane grid         = {img_n}² px, "
        f"pixel_scale={img_pix:.6g}\", width={img_w:.6g}\""
    )
    print(
        f"  mask                     = shape={tuple(mask_2d.shape_native)}, "
        f"pixel_scales={mask_2d.pixel_scales}, "
        f"radius={rec_cfg.get('mask_radius')} arcsec, "
        f"mask_n_pixels={rec_cfg.get('mask_n_pixels')}"
    )
    print(
        f"  source-grid (settings)   = {src_n}² px, "
        f"pixel_scale={src_pix:.6g}\", width={src_w:.6g}\""
    )
    print(f"  transformer              = {getattr(transformer_class, '__name__', transformer_class)}")
    print(f"  use_jax (requested)      = {rec_cfg.get('use_jax', True)}")
    print(
        f"  use_positive_only_solver  = {rec_cfg.get('use_positive_only_solver', False)}"
    )
    print(f"  mesh_type                = {mesh_type}")
    if mesh_type == "delaunay":
        print(
            f"  delaunay                 = image_mesh_shape={rec_cfg.get('image_mesh_shape')}, "
            f"edge_pixels={rec_cfg.get('delaunay_edge_pixels')}, "
            f"areas_factor={rec_cfg.get('delaunay_areas_factor', 0.5)}"
        )
    else:
        print(f"  mesh_shape               = {rec_cfg.get('mesh_shape')}")
    print(f"  regularization.type      = {reg_type}")
    print(f"  regularization prior     = {format_regularization_config(reg_cfg, reg_type)}")
    print(f"  fix_lens                 = {fix_lens}")
    print(
        f"  lens_mass_model centre   = "
        f"centre_0={mass_cfg['centre_0']}, centre_1={mass_cfg['centre_1']}"
    )
    print(
        "  Autolens PowerLaw.centre  = (y, x) = (centre_0, centre_1) "
        "from lens_mass_model when fix_lens=true "
        "(note: KinMS source uses centre_0=x, centre_1=y)"
    )
    print(f"  lens einstein_radius     = {mass_cfg.get('einstein_radius')}")
    print(
        f"  lens ell_comps           = "
        f"({mass_cfg.get('elliptical_comps_0')}, {mass_cfg.get('elliptical_comps_1')})"
    )
    print(f"  lens slope               = {mass_cfg.get('slope')}")
    print(
        f"  lens shear (JSON)        = "
        f"({mass_cfg.get('shear_elliptical_comps_0')}, "
        f"{mass_cfg.get('shear_elliptical_comps_1')}), "
        f"swap_shear_components={settings.get('swap_shear_components', True)}"
    )
    if centre_prior is not None and not fix_lens:
        print(f"  free lens centre_prior   = {centre_prior}")
    if truth:
        print(
            f"  truth source centre      = "
            f"centre_0={truth.get('centre_0')}, centre_1={truth.get('centre_1')}"
        )
    if dataset is not None:
        try:
            n_vis = int(np.asarray(dataset.data).size)
        except Exception:
            n_vis = "?"
        print(f"  moment-0 visibilities    = {n_vis} complex samples")

    print(f"  free parameter count     = {model.prior_count}")
    try:
        # Autofit model tree with fixed values + free priors.
        print("  model.info:")
        for line in str(model.info).splitlines():
            print(f"    {line}")
    except Exception as exc:
        print(f"  model.info unavailable ({exc})")

    try:
        start = model.instance_from_prior_medians()
        lens_c = start.galaxies.lens.mass.centre
        print(
            f"  start instance lens.centre (y, x) = {_fmt_centre(lens_c)}  "
            f"[centre[0]={float(lens_c[0]):.6f}, centre[1]={float(lens_c[1]):.6f}]"
        )
        reg = start.galaxies.source.pixelization.regularization
        if hasattr(reg, "coefficient"):
            print(f"  start regularization coeff  = {reg.coefficient}")
        else:
            print(
                f"  start regularization        = "
                f"inner={reg.inner_coefficient}, outer={reg.outer_coefficient}, "
                f"signal_scale={reg.signal_scale}"
            )
    except Exception as exc:
        print(f"  start instance unavailable ({exc})")
    print("=== End phase-1 setup ===\n")


def print_phase1_result(instance, fit, settings=None):
    """Print the phase-1 best-fit / evaluated instance parameters."""
    mass = instance.galaxies.lens.mass
    lens_c = mass.centre
    reg = instance.galaxies.source.pixelization.regularization
    print("\n=== Phase-1 parameters (result) ===")
    print(
        f"  lens.mass.centre         = {_fmt_centre(lens_c)}  "
        f"[Autolens (y, x) = (centre[0], centre[1])]"
    )
    if settings is not None:
        mass_cfg = settings["lens_mass_model"]
        print(
            f"  JSON lens centre_0/1     = "
            f"({mass_cfg['centre_0']}, {mass_cfg['centre_1']})  "
            f"[passed to Autolens as (y, x)=(centre_0, centre_1)]"
        )
        print(
            f"  Δ lens (instance − JSON) = "
            f"({float(lens_c[0]) - float(mass_cfg['centre_0']):+.6f}, "
            f"{float(lens_c[1]) - float(mass_cfg['centre_1']):+.6f})"
        )
        truth = settings.get("truth_parameters") or {}
        if truth and "centre_0" in truth and "centre_1" in truth:
            tx = float(truth["centre_0"])
            ty = float(truth["centre_1"])
            print(
                f"  truth source centre      = "
                f"centre_0={tx} (x), centre_1={ty} (y)  "
                f"→ Autolens (y, x)=({ty}, {tx})"
            )
    print(f"  lens einstein_radius     = {getattr(mass, 'einstein_radius', None)}")
    print(f"  lens ell_comps           = {getattr(mass, 'ell_comps', None)}")
    print(f"  lens slope               = {getattr(mass, 'slope', None)}")
    if hasattr(reg, "coefficient"):
        print(f"  regularization coeff     = {reg.coefficient}")
    else:
        print(
            f"  regularization           = "
            f"inner={reg.inner_coefficient}, outer={reg.outer_coefficient}, "
            f"signal_scale={reg.signal_scale}"
        )
    try:
        mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]
        mesh = _to_numpy(mapper.source_plane_mesh_grid)
        print(
            f"  source-plane mesh bbox   = "
            f"y=[{mesh[:, 0].min():.4f}, {mesh[:, 0].max():.4f}], "
            f"x=[{mesh[:, 1].min():.4f}, {mesh[:, 1].max():.4f}] arcsec  "
            f"({mesh.shape[0]} pixels)"
        )
        recon = _to_numpy(fit.inversion.reconstruction)
        peak = int(np.argmax(recon))
        py, px = float(mesh[peak, 0]), float(mesh[peak, 1])
        print(
            f"  source recon peak        = "
            f"Autolens (y, x)=({py:.4f}, {px:.4f})  "
            f"→ LensKin (x, y)=({px:.4f}, {py:.4f})  "
            f"(value={recon[peak]:.6g})"
        )
        if settings is not None:
            truth = settings.get("truth_parameters") or {}
            if truth and "centre_0" in truth and "centre_1" in truth:
                tx = float(truth["centre_0"])
                ty = float(truth["centre_1"])
                # Compare in Autolens (y, x)
                dy = py - ty
                dx = px - tx
                dist = float(np.hypot(dx, dy))
                print(
                    f"  Δ peak − truth (y, x)    = "
                    f"({dy:+.4f}, {dx:+.4f}) arcsec  "
                    f"|Δ|={dist:.4f} arcsec"
                )
        try:
            dirty = _to_numpy(fit.dirty_image)
            # Prefer native 2D array if present
            if dirty.ndim > 2:
                dirty = np.asarray(dirty).reshape(-1, dirty.shape[-1])
            if hasattr(fit, "dataset") and hasattr(fit.dataset, "grid"):
                grid = _to_numpy(fit.dataset.grid)
                if grid.ndim == 2 and grid.shape[1] == 2 and dirty.size == grid.shape[0]:
                    ip = int(np.nanargmax(np.abs(dirty)))
                    print(
                        f"  dirty-image peak         = "
                        f"Autolens (y, x)=({float(grid[ip, 0]):.4f}, "
                        f"{float(grid[ip, 1]):.4f})"
                    )
        except Exception as exc:
            print(f"  dirty-image peak unavailable ({exc})")
    except Exception as exc:
        print(f"  source mesh summary unavailable ({exc})")
    print("=== End phase-1 result ===\n")


def run_reconstruction(settings):
    from src.pipelines.runner import load_cube_data, load_cube_data_weights

    apply_lensing_off_reconstruction_overrides(settings)

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    n_pix, pix_scale, fov = autolens_utils.resolve_image_plane_grid_in_settings(
        settings, uv_wavelengths
    )
    logger.info(
        "Image-plane grid: %s², pixel_scale=%.5f arcsec, FOV=%.4f arcsec "
        "(Nyquist 0.5 λ/b_max when pixel_scale is auto/nyquist)",
        n_pix,
        pix_scale,
        fov,
    )
    mask_2d = reconstruction_mask_from_settings(settings)
    rec_cfg = settings["reconstruction"]
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")
    moment0_settings = moment0_settings_from_reconstruction(rec_cfg)
    weights = load_cube_data_weights(settings)
    transformer_class = autolens_utils.transformer_class_from_settings(settings)

    from src.utils.primary_beam import primary_beam_from_settings
    frequencies_arr = np.squeeze(np.asarray(frequencies, dtype=float))
    if np.nanmax(frequencies_arr) < 1.0e4:
        frequencies_hz = frequencies_arr * 1.0e9
    else:
        frequencies_hz = frequencies_arr
    pb = primary_beam_from_settings(settings, frequencies_hz, mask_2d)
    if pb is not None:
        transformer_class = autolens_utils.transformer_class_with_primary_beam(
            transformer_class, pb
        )
        logger.info("Phase-1 primary beam enabled (min=%.6f)", pb.min())

    logger.info(
        "Phase-1 transformer: %s",
        getattr(transformer_class, "__name__", transformer_class),
    )
    dataset = moment0_dataset_from(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        mask_2d=mask_2d,
        transformer_class=transformer_class,
        moment0_settings=moment0_settings,
        weights=weights,
    )
    logger.info(
        "Phase-1 moment0: collapse_mode=%s, sigma_mode=%s, sigma_scale=%s, uv_mode=%s",
        moment0_settings["collapse_mode"],
        moment0_settings["sigma_mode"],
        moment0_settings["sigma_scale"],
        moment0_settings["uv_mode"],
    )

    use_jax = rec_cfg.get("use_jax", True)
    if mesh_type == "delaunay" and use_jax:
        logger.warning(
            "Phase-1 Delaunay: disabling JAX sparse operator (use scipy triangulation). "
            "Interferometer NUFFT can still use JAX via transformer_class."
        )
        use_jax = False
    from src.utils.jax_compat import ensure_numpy_jax_stub, jax_is_usable

    ensure_numpy_jax_stub()
    if use_jax and not jax_is_usable():
        logger.warning(
            "Phase-1: jax/jaxlib unavailable on this host; forcing use_jax=False."
        )
        use_jax = False
    if use_jax:
        dataset = dataset.apply_sparse_operator(use_jax=True)

    adapt_images = adapt_images_for_reconstruction(
        settings=settings,
        mask_2d=mask_2d,
        dataset=dataset,
    )
    mesh_type = settings["reconstruction"].get(
        "mesh_type", "rectangular_adapt_density"
    )
    reg_type = regularization_type_from_settings(rec_cfg)
    logger.info(
        "Phase-1 mesh: type=%s, regularization=%s, mask=%s px, mesh_shape=%s, fix_lens=%s",
        mesh_type,
        reg_type,
        mask_2d.shape_native[0],
        settings["reconstruction"].get("mesh_shape"),
        settings["reconstruction"].get("fix_lens", False),
    )
    model = build_reconstruction_model(settings, mask_2d=mask_2d)
    print_phase1_setup(
        settings=settings,
        mask_2d=mask_2d,
        model=model,
        dataset=dataset,
        transformer_class=transformer_class,
    )
    logger.info(
        "Phase-1 free parameters (%s): %s",
        model.prior_count,
        model.info if hasattr(model, "info") else "see model.info",
    )
    search_cfg = rec_cfg.get("search", {})
    search = build_optimizer_from_settings(search_cfg, mesh_type=mesh_type)

    from src.pipelines.phase1_analysis import Phase1AnalysisInterferometer

    figure_of_merit = search_cfg.get(
        "figure_of_merit", "log_likelihood_with_regularization"
    )
    logger.info("Phase-1 optimizer figure_of_merit: %s", figure_of_merit)

    use_positive_only_solver = use_positive_only_solver_from_settings(settings)
    logger.info("Phase-1 use_positive_only_solver: %s", use_positive_only_solver)

    analysis = Phase1AnalysisInterferometer(
        dataset=dataset,
        adapt_images=adapt_images,
        settings=al.Settings(use_positive_only_solver=use_positive_only_solver),
        raise_inversion_positions_likelihood_exception=False,
        use_jax=use_jax,
        figure_of_merit=figure_of_merit,
    )

    # Phase 1 is a preliminary internal step whose results are extracted
    # programmatically, so skip the (interferometer) dataset visualization,
    # which can fail on the moment-0 dataset for some matplotlib versions.
    if not settings["reconstruction"].get("visualize", False):
        analysis.should_visualize = _never_visualize

    from src.pipelines.phase1_likelihood import (
        format_likelihood_breakdown,
        phase1_likelihood_breakdown,
    )

    # fix_lens=true + fixed regularization => zero free parameters. Autofit
    # cannot run a search on a 0-D model; evaluate the fixed reconstruction once.
    if model.prior_count == 0:
        logger.info(
            "Phase-1 model has no free parameters "
            "(fixed lens + fixed regularization); evaluating once."
        )
        instance = model.instance_from_prior_medians()
        fit = analysis.fit_from(instance=instance)
        result = _FixedPhase1Result(instance=instance, fit=fit)
    else:
        result = search.fit(model=model, analysis=analysis)
        instance = result.max_log_likelihood_instance
        fit = result.max_log_likelihood_fit

    print_phase1_result(instance=instance, fit=fit, settings=settings)

    reg = instance.galaxies.source.pixelization.regularization
    breakdown = phase1_likelihood_breakdown(fit)
    if hasattr(reg, "coefficient"):
        logger.info("Phase-1 regularization coefficient: %s", reg.coefficient)
    else:
        logger.info(
            "Phase-1 regularization: inner=%s outer=%s signal_scale=%s",
            reg.inner_coefficient,
            reg.outer_coefficient,
            reg.signal_scale,
        )
    logger.info("Phase-1 likelihood breakdown: %s", format_likelihood_breakdown(breakdown))
    return result


def _to_numpy(obj):
    """
    Convert an autoarray / JAX-backed structure to a plain NumPy array.

    autoarray structures wrap their data (which may be a JAX array) and their
    ``__array__`` returns that inner object directly, which newer NumPy rejects.
    Unwrapping one level via ``.array`` and then converting handles both the
    NumPy and JAX cases.
    """
    obj = getattr(obj, "array", obj)
    return np.asarray(obj)


PHASE2_SOURCE_N_PIXELS = autolens_utils.DEFAULT_KINMS_SOURCE_N_PIXELS


def source_plane_bounding_box_from_result(result):
    """
    Bounding box [y_min, y_max, x_min, x_max] of the phase-1 pixelization mesh
    on the lensed source plane (arcsec).
    """
    fit = result.max_log_likelihood_fit
    mapper = fit.inversion.cls_list_from(cls=al.Mapper)[0]
    mesh_grid = _to_numpy(mapper.source_plane_mesh_grid)

    return np.array(
        [
            mesh_grid[:, 0].min(),
            mesh_grid[:, 0].max(),
            mesh_grid[:, 1].min(),
            mesh_grid[:, 1].max(),
        ],
        dtype=float,
    )


def phase2_source_grid_from_result(result, n_channels, settings=None):
    """
    Build the phase-2 KinMS source grid on the phase-1 Autolens source-plane extent.

    The bounding box is taken from the phase-1 pixelization mesh. Resolution is
    ``source_grid.n_pixels`` in settings (default 256).
    """
    bounding_box = source_plane_bounding_box_from_result(result)
    if settings is None:
        n_pixels = PHASE2_SOURCE_N_PIXELS
    else:
        n_pixels = autolens_utils.kinms_source_n_pixels_from_settings(settings)
    return Grid3D.bounding_box(
        bounding_box=bounding_box,
        n_pixels=n_pixels,
        n_channels=n_channels,
    )


def _scalar_pixel_scale(pixel_scales):
    """Return a single float pixel scale from a scalar or (y, x) pair."""
    ps = np.asarray(pixel_scales, dtype=float).ravel()
    if ps.size == 0:
        raise ValueError("pixel_scales is empty")
    return float(ps[0])


def _grid_2d_pixel_scale(grid_2d):
    """Pixel scale (arcsec) of an Autolens / LensKin 2D grid."""
    if hasattr(grid_2d, "pixel_scales"):
        return _scalar_pixel_scale(grid_2d.pixel_scales)
    if hasattr(grid_2d, "pixel_scale"):
        return float(grid_2d.pixel_scale)
    raise ValueError(
        "grid_2d must expose pixel_scales or pixel_scale to convert "
        "Autolens Jy/image-pixel units onto the destination grid."
    )


def rescale_sb_from_image_pixels(sb_map, image_pixel_scale, grid_pixel_scale):
    """
    Convert a map in Jy per **image-plane** pixel to Jy per **grid** pixel.

    Autolens interferometer reconstructions are stored in the same Jy/pixel
    units as the image-plane mask. When the destination source grid uses a
    different pixel area, each value must be scaled by
    ``(grid_pixel_scale / image_pixel_scale)^2`` before summing for KinMS /
    GalPaK ``intFlux``. Omitting this factor overcounts flux whenever the
    phase-2 grid is finer than the image grid (typical for mesh-bbox grids).
    """
    image_ps = float(image_pixel_scale)
    grid_ps = float(grid_pixel_scale)
    if not np.isfinite(image_ps) or image_ps <= 0.0:
        raise ValueError(f"image_pixel_scale must be positive (got {image_pixel_scale!r})")
    if not np.isfinite(grid_ps) or grid_ps <= 0.0:
        raise ValueError(f"grid_pixel_scale must be positive (got {grid_pixel_scale!r})")
    return np.asarray(sb_map, dtype=float) * (grid_ps / image_ps) ** 2


def interpolate_reconstruction_to_grid(
    reconstruction,
    source_plane_mesh_grid,
    grid_2d,
):
    """
    Interpolate an inversion reconstruction onto a regular ``grid_2d``.

    Autolens source-pixel values are **surface brightness in Jy per image-plane
    pixel** (not flux per mesh cell, and not yet Jy per ``grid_2d`` pixel).
    Linear interpolation onto a finer regular grid must **not** force
    ``sum(sb_map) == sum(reconstruction)``: for a uniform mesh covering the
    same area as an ``N``-times finer grid, the sum grows by about ``N²`` while
    values remain in image-pixel units.

    Callers that need Jy per destination pixel (KinMS / GalPaK normalization)
    must then apply :func:`rescale_sb_from_image_pixels`.
    """
    reconstruction = np.asarray(reconstruction, dtype=float).ravel()
    source_plane_mesh_grid = np.asarray(source_plane_mesh_grid, dtype=float)
    interpolation_coords = np.asarray(grid_2d, dtype=float)

    if source_plane_mesh_grid.ndim != 2 or source_plane_mesh_grid.shape[1] != 2:
        raise ValueError(
            "source_plane_mesh_grid must have shape (n_mesh, 2); "
            f"got {source_plane_mesh_grid.shape}"
        )
    if reconstruction.shape[0] != source_plane_mesh_grid.shape[0]:
        raise ValueError(
            "reconstruction length must match mesh points: "
            f"{reconstruction.shape[0]} vs {source_plane_mesh_grid.shape[0]}"
        )

    interpolated = griddata(
        points=source_plane_mesh_grid,
        values=reconstruction,
        xi=interpolation_coords,
        fill_value=0.0,
    )
    return np.asarray(interpolated.reshape(grid_2d.shape_native), dtype=float)


def source_sb_from_fit(fit, grid_2d):
    """
    Interpolate a ``FitInterferometer`` source reconstruction onto ``grid_2d``.

    Returns a map in Jy per ``grid_2d`` pixel (per channel), converting from
    Autolens image-plane Jy/pixel units via :func:`rescale_sb_from_image_pixels`.
    """
    inversion = fit.inversion
    mapper = inversion.cls_list_from(cls=al.Mapper)[0]

    sb_image_units = interpolate_reconstruction_to_grid(
        reconstruction=_to_numpy(inversion.reconstruction),
        source_plane_mesh_grid=_to_numpy(mapper.source_plane_mesh_grid),
        grid_2d=grid_2d,
    )
    image_pixel_scale = _scalar_pixel_scale(fit.dataset.mask.pixel_scales)
    grid_pixel_scale = _grid_2d_pixel_scale(grid_2d)
    return rescale_sb_from_image_pixels(
        sb_image_units,
        image_pixel_scale=image_pixel_scale,
        grid_pixel_scale=grid_pixel_scale,
    )


def source_sb_on_grid(result, grid_2d):
    """
    Interpolate the pixelized source reconstruction onto ``grid_2d`` and return
    the surface-brightness map plus the best-fit lens centre from phase 1.

    For phase 2, ``grid_2d`` should be the dedicated KinMS source grid returned
    by :func:`phase2_source_grid_from_result`. The pixelized mesh resolution is
    set separately via ``reconstruction.mesh_shape``.
    """
    fit = result.max_log_likelihood_fit
    sb_map = source_sb_from_fit(fit, grid_2d)

    instance = result.max_log_likelihood_instance
    mass_centre = instance.galaxies.lens.mass.centre
    centre = (float(mass_centre[0]), float(mass_centre[1]))

    return sb_map, centre


DEFAULT_FLUX_SNR_THRESHOLD = 0.5


def flux_snr_threshold_from_settings(settings, default=DEFAULT_FLUX_SNR_THRESHOLD):
    """
    SNR cut applied to the phase-1 source map before locking total flux.

    Read from ``reconstruction.flux_snr_threshold``. Default ``0.5``. Set to
    ``null``, ``false``, or ``<= 0`` to disable (sum the full map).
    """
    rec_cfg = (settings or {}).get("reconstruction") or {}
    if "flux_snr_threshold" not in rec_cfg:
        if default is None:
            return None
        thr = float(default)
        return None if thr <= 0.0 else thr

    value = rec_cfg["flux_snr_threshold"]
    if value is None or value is False:
        return None
    thr = float(value)
    if thr <= 0.0:
        return None
    return thr


def source_noise_on_grid(result, grid_2d):
    """
    Interpolate Autolens ``inversion.reconstruction_noise_map`` onto ``grid_2d``.

    Uses the same Jy/image-pixel → Jy/grid-pixel rescale as
    :func:`source_sb_from_fit`, so ``sb / noise`` is dimensionless SNR.
    """
    fit = result.max_log_likelihood_fit
    inversion = fit.inversion
    mapper = inversion.cls_list_from(cls=al.Mapper)[0]
    noise_image_units = interpolate_reconstruction_to_grid(
        reconstruction=_to_numpy(inversion.reconstruction_noise_map),
        source_plane_mesh_grid=_to_numpy(mapper.source_plane_mesh_grid),
        grid_2d=grid_2d,
    )
    return rescale_sb_from_image_pixels(
        noise_image_units,
        image_pixel_scale=_scalar_pixel_scale(fit.dataset.mask.pixel_scales),
        grid_pixel_scale=_grid_2d_pixel_scale(grid_2d),
    )


def apply_reconstruction_snr_mask(sb_map, noise_map, snr_threshold):
    """
    Zero pixels with ``sb / noise < snr_threshold``.

    Negative SB yields negative SNR and is removed for any positive threshold.
    If ``snr_threshold`` is ``None``, return ``sb_map`` unchanged.
    """
    sb = np.asarray(sb_map, dtype=float)
    if snr_threshold is None:
        return sb
    thr = float(snr_threshold)
    if thr <= 0.0:
        return sb
    noise = np.asarray(noise_map, dtype=float)
    if noise.shape != sb.shape:
        raise ValueError(
            f"noise_map shape {noise.shape} must match sb_map shape {sb.shape}"
        )
    snr = np.zeros_like(sb)
    good = noise > 0.0
    snr[good] = sb[good] / noise[good]
    return np.where(snr >= thr, sb, 0.0)


def source_sb_for_phase2_flux(result, grid_2d, snr_threshold=DEFAULT_FLUX_SNR_THRESHOLD):
    """
    Phase-1 SB on ``grid_2d``, optionally SNR-masked for total-flux locking.

    Returns ``(sb_masked, centre, sb_full, noise_map)``. ``sb_full`` / ``noise_map``
    are useful for diagnostics; ``sb_masked`` is what should feed
    ``kinms_intflux_from_sb_map`` / pixelized cloudlets when a threshold is set.
    """
    sb_full, centre = source_sb_on_grid(result, grid_2d)
    if snr_threshold is None or float(snr_threshold) <= 0.0:
        return sb_full, centre, sb_full, None
    noise_map = source_noise_on_grid(result, grid_2d)
    sb_masked = apply_reconstruction_snr_mask(sb_full, noise_map, snr_threshold)
    return sb_masked, centre, sb_full, noise_map
