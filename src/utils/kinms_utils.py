import os, sys
import numpy as np
import scipy.integrate

# NOTE:
def _ensure_scipy_cumtrapz():
    """KinMS uses scipy.integrate.cumtrapz, removed in SciPy >= 1.14."""
    if hasattr(scipy.integrate, "cumtrapz"):
        return
    from scipy.integrate import cumulative_trapezoid

    def cumtrapz(y, x, initial=0):
        out = cumulative_trapezoid(y, x, initial=0)
        if initial:
            return np.concatenate([[initial], out])
        return out

    scipy.integrate.cumtrapz = cumtrapz


def _ensure_numpy_product():
    """KinMS uses np.product, removed in NumPy 2.0."""
    if not hasattr(np, "product"):
        np.product = np.prod


def _patch_kinms_numpy_compat():
    """
    Patch KinMS for NumPy 2 / recent SciPy.

  KinMS tests ``if np.any(flux_clouds) != None``, which parses as
  ``(np.any(flux_clouds)) != None``. When ``flux_clouds`` is ``None``,
  ``np.any(None)`` is ``False`` and ``False != None`` is ``True``, so
  parametric ``sbProf`` models always take the ``flux_clouds`` branch and
  raise (or mis-normalise). Pixelized ``inClouds`` mode is unaffected.
  """
    from kinms import KinMS

    if getattr(KinMS, "_lenskin_numpy_compat_patched", False):
        return
    KinMS._lenskin_numpy_compat_patched = True

    def histo_with_bincount(self, vals, bins):
        cd = vals[:, 2]
        cd += bins[2] * vals[:, 1]
        cd += (bins[2] * bins[1]) * vals[:, 0]
        return np.bincount(cd.astype(int), minlength=np.prod(bins)).reshape(
            *bins
        ).astype(np.float64)

    KinMS.histo_with_bincount = histo_with_bincount

    def add_fluxes(self, clouds2do, subs):
        if not getattr(self, "inClouds_given", False):
            self.flux_clouds = None

        nsubs = subs.sum()
        if nsubs > 0:
            if self.flux_clouds is not None and self.inClouds_given:
                cube, _edges = np.histogramdd(
                    clouds2do,
                    bins=(self.x_size, self.y_size, self.v_size),
                    range=(
                        (0, self.x_size),
                        (0, self.y_size),
                        (0, self.v_size),
                    ),
                    weights=self.flux_clouds[subs],
                )
            else:
                cube = self.histo_with_bincount(
                    clouds2do,
                    bins=np.array([self.x_size, self.y_size, self.v_size]),
                )
        else:
            cube = np.zeros(
                (int(self.x_size), int(self.y_size), int(self.v_size))
            )
        return cube

    KinMS.add_fluxes = add_fluxes

    def normalise_cube(self, cube, psf):
        if self.intFlux > 0:
            if not self.cleanOut:
                cube *= (self.intFlux * psf.sum()) / (cube.sum() * self.dv)
            else:
                cube *= self.intFlux / (cube.sum() * self.dv)
        elif self.flux_clouds is not None and getattr(
            self, "inClouds_given", False
        ):
            cube *= self.flux_clouds.sum() / cube.sum()
        else:
            cube /= cube.sum()
        return cube

    KinMS.normalise_cube = normalise_cube


_ensure_scipy_cumtrapz()
_ensure_numpy_product()

try:
    from kinms import KinMS

    _patch_kinms_numpy_compat()
except ImportError:
    print("\'kinMS\' could not be imported")
    KinMS = None

# NOTE:
from src.grid.grid import (
    Grid3D,
)


def kinms_radial_grid(pixel_scale: float, field_width_arcsec: float, n_points: int = 10000):
    """
    Log-spaced radial grid for KinMS ``sbRad`` / ``velRad``.

    Matches ``generate_kinms_lensed_cube.kinms_radial_grid`` (mock generation).
    """
    log_xmin = np.log10(pixel_scale / 5.0)
    log_xmax = np.log10(field_width_arcsec)
    return np.logspace(log_xmin, log_xmax, n_points)


class InstanceKinMS:

    def __init__(
        self,
        obj,
        x=None,
        inClouds=None,
        flux_clouds=None,
        int_flux=None,
        grid_3d=None,
        max_radius=None,
        disk_thick=None,
    ):

        self.obj = obj

        # NOTE:
        if x is None:
            field_width = max(self.obj.xs, self.obj.ys)
            self.x = kinms_radial_grid(self.obj.cellSize, field_width)
        else:
            self.x = x

        self.inClouds = inClouds
        self.flux_clouds = flux_clouds
        self.int_flux = int_flux
        self.grid_3d = grid_3d
        self.max_radius = max_radius
        self.disk_thick = disk_thick

    @property
    def instance(self):
        return self.obj



# NOTE:
def make_kinms_instance(
    n_pixels: int,
    pixel_scale: float,
    n_channels: int,
    z_step_kms: float,
    nSamps=5e6,
    field_width_arcsec=None,
):
    if field_width_arcsec is None:
        field_width = n_pixels * pixel_scale
        return KinMS(
            xs=field_width,
            ys=field_width,
            vs=n_channels * z_step_kms,
            cellSize=pixel_scale,
            dv=z_step_kms,
            beamSize=None,
            cleanOut=True,
            nSamps=nSamps,
        )

    field_width_y, field_width_x = field_width_arcsec
    # KinMS uses a single cellSize for both axes, so non-square fields can return
    # a model cube with a different shape than ``n_pixels``. The cube is
    # resampled onto ``grid_3d`` before ray tracing.
    return KinMS(
        xs=field_width_x,
        ys=field_width_y,
        vs=n_channels * z_step_kms,
        cellSize=field_width_x / n_pixels,
        dv=z_step_kms,
        beamSize=None,
        cleanOut=True,
        nSamps=nSamps,
    )


# # NOTE:
# def make_instance_from_masked_dataset(
#     masked_dataset,
#     nSamps=5e6,
# ):
#
#     return make_kinms_instance(
#         n_pixels=masked_dataset.n_pixels,
#         pixel_scale=masked_dataset.pixel_scale,
#         n_channels=masked_dataset.n_channels,
#         z_step_kms=masked_dataset.z_step_kms,
#         nSamps=nSamps,
#     )

def make_instance_from_grid(
    grid_3d: Grid3D,
    z_step_kms: float,
    nSamps=5e6,
    attach_grid: bool = False,
    disk_thick=None,
):
    field_width_arcsec = _field_width_arcsec_from_grid(grid_3d)
    pixel_scales = grid_3d.pixel_scales
    if isinstance(pixel_scales, (int, float)):
        pixel_scale = pixel_scales
    else:
        pixel_scale = pixel_scales[1]

    obj = make_kinms_instance(
        n_pixels=grid_3d.n_pixels,
        pixel_scale=pixel_scale,
        n_channels=grid_3d.n_channels,
        z_step_kms=z_step_kms,
        nSamps=nSamps,
        field_width_arcsec=field_width_arcsec,
    )

    instance = InstanceKinMS(obj=obj, disk_thick=disk_thick)
    if attach_grid:
        instance.grid_3d = grid_3d
    return instance


def integrated_sb_per_pixel(
    sb_map: np.ndarray,
    z_step_kms: float,
    n_channels: int,
    input_units: str = "jy_per_pixel_per_channel",
) -> np.ndarray:
    """
    Convert a 2D surface-brightness map to velocity-integrated units.

    Returns a map in Jy km/s/pixel, matching the moment-0 convention used in
    ``pixelized_plots`` (``sum_over_channels * z_step_kms``).

    Parameters
    ----------
    sb_map
        2D surface-brightness map on the KinMS grid.
    z_step_kms
        Velocity channel width in km/s.
    n_channels
        Number of spectral channels in the data cube.
    input_units
        ``jy_per_pixel_per_channel`` — map is in Jy/pixel per channel (e.g. the
        velocity-averaged phase-1 reconstruction). Integrate over velocity as
        ``sb_map * n_channels * z_step_kms``.
        ``jy_kms_per_pixel`` — map is already in Jy km/s/pixel (e.g. moment-0).
    """
    if input_units == "jy_per_pixel_per_channel":
        return sb_map * n_channels * z_step_kms
    if input_units == "jy_kms_per_pixel":
        return sb_map
    raise ValueError(
        f"Unsupported sb_input_units: {input_units!r}. "
        "Expected 'jy_per_pixel_per_channel' or 'jy_kms_per_pixel'."
    )


def kinms_intflux_from_sb_map(
    sb_map,
    z_step_kms: float,
    n_channels: int,
    sb_input_units: str = "jy_per_pixel_per_channel",
):
    """
    Return the total velocity-integrated source flux for KinMS ``intFlux``.

    KinMS documents ``intFlux`` in Jy/km/s. After normalization with
    ``cleanOut=True``, the model cube satisfies ``cube.sum() * dv == intFlux``.

    Phase-1 maps from :func:`reconstruction.source_sb_on_grid` are in Jy/pixel
    per channel (velocity-averaged visibilities). They are converted to Jy
    km/s/pixel before summing over the source grid.
    """
    sb_integrated = integrated_sb_per_pixel(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=n_channels,
        input_units=sb_input_units,
    )
    return float(sb_integrated.sum())


def total_flux_from_sb_map(
    sb_map,
    z_step_kms: float,
    n_channels: int,
    sb_input_units: str = "jy_per_pixel_per_channel",
):
    """Alias for :func:`kinms_intflux_from_sb_map`."""
    return kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=n_channels,
        sb_input_units=sb_input_units,
    )


def in_clouds_and_flux_from_sb_map(
    sb_map,
    grid_2d,
    z_step_kms: float,
    n_channels: int,
    flux_threshold=0.0,
    sb_input_units: str = "jy_per_pixel_per_channel",
    clouds_per_pixel: int = 1,
    scale_height_arcsec: float = 0.0,
    seed: int = 100,
):
    """
    Build KinMS inClouds (arcsec) and flux weights from a regular-grid SB map.

    Grid coordinates follow autolens (y, x); KinMS inClouds use (x, y, z).

    Coordinates are returned in **absolute** source-plane arcsec (the same
    frame as ``grid_2d``). ``kinMSPixelized.make_model`` subtracts the free
    source ``centre`` before calling KinMS, because KinMS expects ``inClouds``
    relative to the phase centre and places that centre with ``phaseCent``.

    Each SB pixel contributes ``clouds_per_pixel`` clouds, jittered uniformly
    within the pixel, optionally with an exponential vertical scale height
    (KinMS ``diskThick`` convention) to reduce lattice aliasing / striping.

    The input map is converted to velocity-integrated surface brightness
    (Jy km/s/pixel) before being passed to KinMS. ``flux_clouds`` are relative
    weights and the total integrated flux is returned separately for use as
    KinMS ``intFlux`` (matching the parametric ``kinMS`` ``sbProf`` + ``intFlux``
    pattern).
    """
    sb_integrated = integrated_sb_per_pixel(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=n_channels,
        input_units=sb_input_units,
    )

    coords = np.asarray(grid_2d)
    shape_2d = sb_integrated.shape
    coords_2d = coords.reshape(shape_2d + (2,))
    y_arcsec = coords_2d[:, :, 0]
    x_arcsec = coords_2d[:, :, 1]
    flux = np.asarray(sb_integrated, dtype=float)

    if flux_threshold > 0.0:
        keep = flux > flux_threshold
    else:
        keep = flux > 0.0

    x_pix = x_arcsec[keep]
    y_pix = y_arcsec[keep]
    flux_pix = flux[keep]

    total_flux = float(flux_pix.sum())
    clouds_per_pixel = int(clouds_per_pixel)
    if clouds_per_pixel < 1:
        raise ValueError(f"clouds_per_pixel must be >= 1; got {clouds_per_pixel}")

    if x_pix.size == 0:
        empty = np.zeros((0, 3), dtype=float)
        return empty, np.zeros(0, dtype=float), 0.0

    # Infer pixel spacing from the regular grid (prefer unique sorted diffs).
    unique_x = np.unique(coords_2d[0, :, 1])
    unique_y = np.unique(coords_2d[:, 0, 0])
    dx = float(np.median(np.diff(unique_x))) if unique_x.size > 1 else 0.0
    dy = float(np.median(np.diff(unique_y))) if unique_y.size > 1 else 0.0

    n_pix = x_pix.size
    n_clouds = n_pix * clouds_per_pixel
    rng = np.random.RandomState(int(seed))

    # Uniform jitter within each pixel; duplicate parent flux evenly.
    jitter_x = (rng.random_sample(n_clouds) - 0.5) * dx
    jitter_y = (rng.random_sample(n_clouds) - 0.5) * dy
    x_clouds = np.repeat(x_pix, clouds_per_pixel) + jitter_x
    y_clouds = np.repeat(y_pix, clouds_per_pixel) + jitter_y

    scale_height_arcsec = float(scale_height_arcsec)
    if scale_height_arcsec > 0.0:
        # Match KinMS exponential scale-height sampling (diskThick).
        z_clouds = (
            scale_height_arcsec
            * rng.exponential(1.0, n_clouds)
            * rng.choice([-1.0, 1.0], size=n_clouds)
        )
    else:
        z_clouds = np.zeros(n_clouds, dtype=float)

    flux_clouds = np.repeat(flux_pix / clouds_per_pixel, clouds_per_pixel)
    if total_flux > 0.0:
        flux_weights = flux_clouds / total_flux
    else:
        flux_weights = flux_clouds

    in_clouds = np.column_stack((x_clouds, y_clouds, z_clouds))
    return in_clouds, flux_weights, total_flux


def _field_width_arcsec_from_grid(grid_3d: Grid3D):
    coords = np.asarray(grid_3d.grid_2d)
    y_extent = float(coords[:, 0].max() - coords[:, 0].min())
    x_extent = float(coords[:, 1].max() - coords[:, 1].min())
    return y_extent, x_extent


DEFAULT_DISK_SCALE_HEIGHT_KPC = 0.1
DEFAULT_CLOUDS_PER_PIXEL = 1024


def disk_scale_height_kpc_from_settings(settings, default=DEFAULT_DISK_SCALE_HEIGHT_KPC):
    """Fixed disc scale height in kpc (not a free fit parameter)."""
    if settings is None:
        return float(default)
    if "disk_scale_height_kpc" in settings:
        return float(settings["disk_scale_height_kpc"])
    rec_cfg = settings.get("reconstruction", {})
    return float(rec_cfg.get("disk_scale_height_kpc", default))


def disk_scale_height_arcsec_from_settings(settings, default_kpc=DEFAULT_DISK_SCALE_HEIGHT_KPC):
    """
    Convert ``disk_scale_height_kpc`` to source-plane arcsec at ``redshift_source``.

    Uses Astropy Planck15 proper transverse scale (same convention as Autolens).
    """
    from astropy.cosmology import Planck15

    kpc = disk_scale_height_kpc_from_settings(settings, default=default_kpc)
    if settings is None or "redshift_source" not in settings:
        raise ValueError(
            "disk_scale_height_arcsec_from_settings requires settings['redshift_source']."
        )
    redshift = float(settings["redshift_source"])
    arcsec_per_kpc = Planck15.arcsec_per_kpc_proper(redshift).value
    return float(kpc * arcsec_per_kpc)


def clouds_per_pixel_from_settings(settings, default=DEFAULT_CLOUDS_PER_PIXEL):
    """Sub-samples per SB pixel for pixelized KinMS clouds (fixed, not free)."""
    rec_cfg = settings.get("reconstruction", {}) if settings is not None else {}
    return int(rec_cfg.get("clouds_per_pixel", default))


def max_radius_from_settings(settings, default=None):
    """
    Fixed maximum radius (arcsec) for the pixelized KinMS cloud distribution.

    Read from ``reconstruction.max_radius``. Not a free fit parameter.
    ``None`` (default) keeps the full SB map; set a positive value to clip
    clouds outside that radius about the source centre.
    """
    rec_cfg = settings.get("reconstruction", {}) if settings is not None else {}
    value = rec_cfg.get("max_radius", default)
    if value is None:
        return None
    return float(value)


def apply_max_radius_to_clouds(in_clouds, flux_clouds, int_flux, max_radius):
    """
    Keep only clouds within ``max_radius`` of the phase centre.

    ``in_clouds`` must already be centre-relative (KinMS phase-centre frame).
    Flux weights are renormalised; ``int_flux`` is scaled by the kept fraction.
    """
    if max_radius is None:
        return in_clouds, flux_clouds, int_flux

    max_radius = float(max_radius)
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError(f"max_radius must be a positive finite value; got {max_radius!r}")

    in_clouds = np.asarray(in_clouds, dtype=float)
    flux_clouds = np.asarray(flux_clouds, dtype=float)
    radius = np.hypot(in_clouds[:, 0], in_clouds[:, 1])
    keep = radius <= max_radius
    if not np.any(keep):
        raise ValueError(
            f"No pixelized KinMS clouds remain within max_radius={max_radius} arcsec."
        )

    kept_weight = float(flux_clouds[keep].sum())
    total_weight = float(flux_clouds.sum())
    in_clouds = in_clouds[keep]
    flux_clouds = flux_clouds[keep]
    if kept_weight > 0.0:
        flux_clouds = flux_clouds / kept_weight
    if int_flux is not None and total_weight > 0.0:
        int_flux = float(int_flux) * (kept_weight / total_weight)
    return in_clouds, flux_clouds, int_flux


def sky_plane_vlos_kms(
    x_arcsec,
    y_arcsec,
    vel_rad_arcsec,
    vel_prof_kms,
    inclination_deg,
    pos_ang_deg,
    gas_sigma_kms=0.0,
    seed=None,
):
    """
    Line-of-sight velocities for clouds already on the sky plane.

    Phase-1 / truth SB maps give projected (sky) positions. KinMS would
    re-project those positions if ``inc`` is applied to ``inClouds``, which
    double-counts inclination and brightens the model. Instead we keep sky
    positions fixed and only project the circular velocity field.

    Deprojection and ``v_los`` match KinMS conventions (PA = 0 places the
    redshifted side along +y; see ``KinMS.kinms_create_velField_oneSided``):

        v_los = -v_circ(R) * cos(θ_disk) * sin(i)  (+ optional gas dispersion)
    """
    x_sky = np.asarray(x_arcsec, dtype=float)
    y_sky = np.asarray(y_arcsec, dtype=float)
    vel_rad = np.asarray(vel_rad_arcsec, dtype=float)
    vel_prof = np.asarray(vel_prof_kms, dtype=float)

    # Inverse of KinMS position_angle_rotation (ang = pos_ang):
    #   x3 = c*x2 + s*y2,  y3 = -s*x2 + c*y2,  with c,s = cos/sin(90 - PA)
    c = np.cos(np.radians(90.0 - float(pos_ang_deg)))
    s = np.sin(np.radians(90.0 - float(pos_ang_deg)))
    x_p = c * x_sky - s * y_sky
    y_p = s * x_sky + c * y_sky

    # Inverse of KinMS inclination_projection for a thin disk (z ≈ 0):
    #   x2 = x1,  y2 = y1 * cos(i)
    cos_i = float(np.cos(np.radians(float(inclination_deg))))
    sin_i = float(np.sin(np.radians(float(inclination_deg))))
    if abs(cos_i) < 1.0e-6:
        cos_i = 1.0e-6 if cos_i >= 0.0 else -1.0e-6
    x_disk = x_p
    y_disk = y_p / cos_i

    radius = np.hypot(x_disk, y_disk)
    theta = np.arctan2(y_disk, x_disk)
    v_circ = np.interp(radius, vel_rad, vel_prof, left=vel_prof[0], right=vel_prof[-1])
    v_los = -v_circ * np.cos(theta) * sin_i

    gas_sigma = float(gas_sigma_kms)
    if gas_sigma > 0.0:
        rng = np.random.RandomState(None if seed is None else int(seed))
        v_los = v_los + rng.normal(0.0, gas_sigma, size=v_los.shape)

    return v_los


def make_pixelized_instance_from_grid(
    grid_3d: Grid3D,
    z_step_kms: float,
    sb_map: np.ndarray,
    flux_threshold=0.0,
    sb_input_units: str = "jy_per_pixel_per_channel",
    nSamps=5e6,
    max_radius=None,
    clouds_per_pixel: int = DEFAULT_CLOUDS_PER_PIXEL,
    scale_height_arcsec: float = 0.0,
    seed: int = 100,
):
    field_width_arcsec = _field_width_arcsec_from_grid(grid_3d)
    pixel_scales = grid_3d.pixel_scales
    if isinstance(pixel_scales, (int, float)):
        pixel_scale = pixel_scales
    else:
        pixel_scale = pixel_scales[1]

    obj = make_kinms_instance(
        n_pixels=grid_3d.n_pixels,
        pixel_scale=pixel_scale,
        n_channels=grid_3d.n_channels,
        z_step_kms=z_step_kms,
        nSamps=nSamps,
        field_width_arcsec=field_width_arcsec,
    )

    in_clouds, flux_clouds, int_flux = in_clouds_and_flux_from_sb_map(
        sb_map=sb_map,
        grid_2d=grid_3d.grid_2d,
        z_step_kms=z_step_kms,
        n_channels=grid_3d.n_channels,
        flux_threshold=flux_threshold,
        sb_input_units=sb_input_units,
        clouds_per_pixel=clouds_per_pixel,
        scale_height_arcsec=scale_height_arcsec,
        seed=seed,
    )

    return InstanceKinMS(
        obj=obj,
        inClouds=in_clouds,
        flux_clouds=flux_clouds,
        int_flux=int_flux,
        grid_3d=grid_3d,
        max_radius=max_radius,
        disk_thick=scale_height_arcsec,
    )


def pixelized_instance_kwargs_from_settings(settings):
    """Fixed pixelized-cloud options from runner settings."""
    return {
        "max_radius": max_radius_from_settings(settings),
        "clouds_per_pixel": clouds_per_pixel_from_settings(settings),
        "scale_height_arcsec": disk_scale_height_arcsec_from_settings(settings),
    }
