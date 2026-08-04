"""Lens mass model construction for parametric and phase-1 fits."""
import autofit as af
import autolens as al

from src.pipelines.priors import prior_from_cfg

# Fixed identity mass used when ``lensing.enabled`` is false.
_IDENTITY_LENS_MASS = {
    "centre_0": 0.0,
    "centre_1": 0.0,
    "einstein_radius": 0.0,
    "elliptical_comps_0": 0.0,
    "elliptical_comps_1": 0.0,
    "slope": 2.0,
    "shear_elliptical_comps_0": 0.0,
    "shear_elliptical_comps_1": 0.0,
    "multipole_m3_elliptical_comps_0": 0.0,
    "multipole_m3_elliptical_comps_1": 0.0,
    "multipole_m4_elliptical_comps_0": 0.0,
    "multipole_m4_elliptical_comps_1": 0.0,
}

_LENS_OFF_MESH_TYPE = "rectangular_uniform"
_LENS_OFF_COMPATIBLE_REG_TYPES = frozenset({"constant", "adapt"})
_LENS_OFF_REG_REMAP = {
    "constant_split": "constant",
    "adapt_split": "adapt",
    # No rectangular AdaptZeroth; drop the zeroth leg when remapping.
    "adapt_split_zeroth": "adapt",
}


def lensing_enabled(settings):
    """
    Return True when gravitational lensing is active.

    Missing ``lensing.enabled`` defaults to True so existing lensed runners
    keep their previous behaviour. Set ``"lensing": {"enabled": false}`` for
    an identity mass model (θ_E = 0) and a forced regular phase-1 mesh.
    """
    lensing = settings.get("lensing")
    if not isinstance(lensing, dict):
        return True
    if "enabled" not in lensing:
        return True
    return bool(lensing["enabled"])


def identity_lens_mass_model(settings=None):
    """
    Return a zero-deflection ``lens_mass_model`` dict.

    If ``settings`` provides ``lens_mass_model.centre_*``, those centres are
    kept (useful for aligning grids); all mass / shear / multipole amplitudes
    are forced to zero.
    """
    mass = dict(_IDENTITY_LENS_MASS)
    if settings is not None:
        cfg = settings.get("lens_mass_model") or {}
        if "centre_0" in cfg:
            mass["centre_0"] = float(cfg["centre_0"])
        if "centre_1" in cfg:
            mass["centre_1"] = float(cfg["centre_1"])
    return mass


def mass_model_for_settings(settings):
    """Effective ``lens_mass_model`` (identity when lensing is disabled)."""
    if not lensing_enabled(settings):
        return identity_lens_mass_model(settings)
    if "lens_mass_model" not in settings:
        raise KeyError(
            "settings['lens_mass_model'] is required when lensing.enabled is true "
            "(or omitted)."
        )
    return settings["lens_mass_model"]


def free_lens_centre_from_settings(settings):
    """Return True when the fit should optimize lens mass centre."""
    if not lensing_enabled(settings):
        return False
    if "free_lens_centre" in settings:
        return bool(settings["free_lens_centre"])
    rec = settings.get("reconstruction")
    if rec is not None:
        return not rec.get("fix_lens", False)
    return False


def validate_lensing_settings(settings):
    """
    Validate lensing-related settings and apply lens-off reconstruction overrides.

    When ``lensing.enabled`` is false:

    - Rejects an explicit request to free the lens centre.
    - Forces ``reconstruction.mesh_type`` to ``rectangular_uniform``.
    - Keeps ``constant`` / ``adapt`` regularization; remaps Delaunay-only
      ``constant_split`` / ``adapt_split`` to their rectangular-compatible
      counterparts; defaults missing types to ``constant``.
    - Forces ``reconstruction.fix_lens`` to true.
    """
    if lensing_enabled(settings):
        if "lens_mass_model" not in settings:
            raise KeyError(
                "settings['lens_mass_model'] is required when lensing is enabled."
            )
        return True

    if settings.get("free_lens_centre") is True:
        raise ValueError(
            "lensing.enabled is false but free_lens_centre is true. "
            "Disable free_lens_centre (or omit it) when lensing is off."
        )

    rec = settings.get("reconstruction")
    if rec is not None:
        apply_lensing_off_reconstruction_overrides(settings)

    # Ensure a mass block exists for tracer / model builders.
    if "lens_mass_model" not in settings:
        settings["lens_mass_model"] = identity_lens_mass_model(settings)
    return False


def apply_lensing_off_reconstruction_overrides(settings):
    """
    Force a regular rectangular phase-1 mesh when lensing is disabled.

    Mutates ``settings['reconstruction']`` in place. No-op when lensing is on
    or when there is no reconstruction block.

    Regularization may be ``constant`` or ``adapt`` (brightness-weighted).
    Delaunay-only ``*_split`` types are remapped to the non-split equivalent.
    """
    if lensing_enabled(settings):
        return settings
    rec = settings.get("reconstruction")
    if rec is None:
        return settings

    requested_mesh = rec.get("mesh_type", "rectangular_adapt_density")
    reg_cfg = rec.setdefault("regularization", {})
    requested_reg = reg_cfg.get("type")

    if requested_mesh != _LENS_OFF_MESH_TYPE:
        print(
            f"lensing.enabled=false: overriding reconstruction.mesh_type "
            f"{requested_mesh!r} -> {_LENS_OFF_MESH_TYPE!r}"
        )

    if requested_reg is None:
        resolved_reg = "constant"
    elif requested_reg in _LENS_OFF_REG_REMAP:
        resolved_reg = _LENS_OFF_REG_REMAP[requested_reg]
        print(
            f"lensing.enabled=false: remapping reconstruction.regularization.type "
            f"{requested_reg!r} -> {resolved_reg!r} "
            f"(rectangular mesh cannot use *_split)"
        )
    elif requested_reg in _LENS_OFF_COMPATIBLE_REG_TYPES:
        resolved_reg = requested_reg
    else:
        raise ValueError(
            f"lensing.enabled=false: unsupported regularization.type={requested_reg!r}. "
            f"Use one of {sorted(_LENS_OFF_COMPATIBLE_REG_TYPES)} "
            f"(or a *_split type, which is remapped)."
        )

    rec["mesh_type"] = _LENS_OFF_MESH_TYPE
    reg_cfg["type"] = resolved_reg
    rec["fix_lens"] = True
    return settings


def _lens_centre_priors_from_settings(settings, centre_prior_cfg=None):
    """
    Build ``centre_0`` / ``centre_1`` priors for a free lens centre.

    Uses ``settings['lens_priors']`` when present, otherwise a box of width
    ``2 * lens_centre_half_width`` around ``lens_mass_model`` values.
    """
    mass_cfg = mass_model_for_settings(settings)
    c0 = float(mass_cfg["centre_0"])
    c1 = float(mass_cfg["centre_1"])
    half_width = float(settings.get("lens_centre_half_width", 0.05))
    lens_priors = settings.get("lens_priors", {})

    if centre_prior_cfg is not None:
        lo = float(centre_prior_cfg["lower_limit"])
        hi = float(centre_prior_cfg["upper_limit"])
        return (
            af.UniformPrior(lower_limit=lo, upper_limit=hi),
            af.UniformPrior(lower_limit=lo, upper_limit=hi),
        )

    priors = {}
    for key, default_centre in (("centre_0", c0), ("centre_1", c1)):
        if key in lens_priors:
            priors[key] = prior_from_cfg(lens_priors[key])
        else:
            priors[key] = af.UniformPrior(
                lower_limit=default_centre - half_width,
                upper_limit=default_centre + half_width,
            )
    return priors["centre_0"], priors["centre_1"]


def lens_galaxy_model_from_settings(
    settings,
    *,
    free_centre=None,
    centre_prior_cfg=None,
):
    """
    Return an ``af.Model(al.Galaxy, ...)`` for the lens from settings JSON.

    When ``free_centre`` is True, only ``mass.centre_0`` / ``mass.centre_1``
    are free; other mass parameters stay fixed at ``lens_mass_model`` values.

    When ``lensing.enabled`` is false, an identity mass is used and the centre
    is never free.
    """
    if not lensing_enabled(settings):
        free_centre = False
    elif free_centre is None:
        free_centre = free_lens_centre_from_settings(settings)

    mass_cfg = mass_model_for_settings(settings)
    mass_kwargs = dict(
        einstein_radius=mass_cfg["einstein_radius"],
        ell_comps=(mass_cfg["elliptical_comps_0"], mass_cfg["elliptical_comps_1"]),
        slope=mass_cfg["slope"],
    )
    if free_centre:
        centre_0_prior, centre_1_prior = _lens_centre_priors_from_settings(
            settings,
            centre_prior_cfg=centre_prior_cfg,
        )
        mass = af.Model(al.mp.PowerLaw, **mass_kwargs)
        mass.centre_0 = centre_0_prior
        mass.centre_1 = centre_1_prior
    else:
        mass = af.Model(
            al.mp.PowerLaw,
            centre=(mass_cfg["centre_0"], mass_cfg["centre_1"]),
            **mass_kwargs,
        )

    shear_swap = settings.get("swap_shear_components", True)
    gamma_1 = (
        mass_cfg["shear_elliptical_comps_1"]
        if shear_swap
        else mass_cfg["shear_elliptical_comps_0"]
    )
    gamma_2 = (
        mass_cfg["shear_elliptical_comps_0"]
        if shear_swap
        else mass_cfg["shear_elliptical_comps_1"]
    )
    shear = af.Model(al.mp.ExternalShear, gamma_1=gamma_1, gamma_2=gamma_2)

    lens_kwargs = {
        "redshift": settings["redshift_lens"],
        "mass": mass,
        "shear": shear,
    }

    if (
        "multipole_m3_elliptical_comps_0" in mass_cfg
        and "multipole_m3_elliptical_comps_1" in mass_cfg
    ):
        m3 = af.Model(
            al.mp.PowerLawMultipole,
            m=3,
            einstein_radius=mass_cfg["einstein_radius"],
            slope=mass_cfg["slope"],
            multipole_comps=(
                mass_cfg["multipole_m3_elliptical_comps_0"],
                mass_cfg["multipole_m3_elliptical_comps_1"],
            ),
        )
        m3.centre = mass.centre
        lens_kwargs["multipole_m3"] = m3

    if (
        "multipole_m4_elliptical_comps_0" in mass_cfg
        and "multipole_m4_elliptical_comps_1" in mass_cfg
    ):
        m4 = af.Model(
            al.mp.PowerLawMultipole,
            m=4,
            einstein_radius=mass_cfg["einstein_radius"],
            slope=mass_cfg["slope"],
            multipole_comps=(
                mass_cfg["multipole_m4_elliptical_comps_0"],
                mass_cfg["multipole_m4_elliptical_comps_1"],
            ),
        )
        m4.centre = mass.centre
        lens_kwargs["multipole_m4"] = m4

    return af.Model(al.Galaxy, **lens_kwargs)


def lens_centre_from_instance(instance, settings):
    """Read lens mass centre from a fit instance, or fall back to settings."""
    if hasattr(instance, "galaxies") and hasattr(instance.galaxies, "lens"):
        mass_centre = instance.galaxies.lens.mass.centre
        return float(mass_centre[0]), float(mass_centre[1])
    mass_cfg = mass_model_for_settings(settings)
    return float(mass_cfg["centre_0"]), float(mass_cfg["centre_1"])
