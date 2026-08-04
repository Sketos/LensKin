import math

PARAMETRIC = "parametric"
PARAMETRIC_FLUX_FROM_PHASE1 = "parametric_flux_from_phase1"
PIXELIZED = "pixelized"

VALID_MODES = {
    PARAMETRIC,
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
}


def normalization_mode_from_settings(settings):
    """
    Return the source normalization mode for a KinMS / GalPaK fit.

    When ``normalization_mode`` is omitted, infer from legacy settings:
    ``model_name: KinMSPixelized`` -> pixelized; otherwise parametric.
    """
    if "normalization_mode" in settings:
        mode = settings["normalization_mode"]
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported normalization_mode: {mode!r}. "
                f"Expected one of {sorted(VALID_MODES)}."
            )
        return mode

    if settings.get("model_name") == "KinMSPixelized":
        return PIXELIZED
    return PARAMETRIC


def requires_phase1(mode):
    return mode in {PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED}


def validate_normalization_settings(settings):
    mode = normalization_mode_from_settings(settings)
    model_name = settings["model_name"]

    if requires_phase1(mode) and "reconstruction" not in settings:
        raise ValueError(
            f"normalization_mode='{mode}' requires a 'reconstruction' block "
            "in the settings file."
        )

    if mode == PARAMETRIC and model_name not in ("KinMS", "GalPak"):
        raise ValueError(
            "normalization_mode='parametric' requires model_name='KinMS' or "
            "'GalPak'."
        )

    if mode == PARAMETRIC_FLUX_FROM_PHASE1 and model_name not in ("KinMS", "GalPak"):
        raise ValueError(
            "normalization_mode='parametric_flux_from_phase1' requires "
            "model_name='KinMS' or 'GalPak'."
        )

    if mode == PIXELIZED and model_name not in ("KinMS", "KinMSPixelized"):
        raise ValueError(
            "normalization_mode='pixelized' requires model_name='KinMS' or "
            "'KinMSPixelized' (GalPaK pixelized is not supported yet)."
        )

    return mode


def intensity_from_phase1_intflux(model_name, intflux_jy_kms, z_step_kms):
    """
    Convert phase-1 velocity-integrated flux to the source model's intensity.

    KinMS ``intensity`` / ``intFlux`` is in Jy km/s (``cube.sum() * dv``).
    GalPaK ``intensity`` / ``flux`` normalizes so ``cube.sum()`` equals flux,
    so the equivalent value is ``intFlux / dv``.
    """
    intflux = float(intflux_jy_kms)
    if model_name == "GalPak":
        dv = float(z_step_kms)
        if not math.isfinite(dv) or dv <= 0.0:
            raise ValueError(
                f"GalPaK flux conversion requires a positive z_step_kms "
                f"(got {z_step_kms!r})."
            )
        return intflux / dv
    return intflux


def priors_for_normalization_mode(priors_cfg, mode, total_flux=None, settings=None):
    """
    Return priors for phase 2, fixing source intensity when required.

    When ``settings['phase2']['kinematics_from_truth']`` is true, kinematic
    parameters are fixed at truth values except those listed in
    ``phase2['free_parameters']``.
    """
    priors = dict(priors_cfg)
    if mode == PARAMETRIC_FLUX_FROM_PHASE1:
        if total_flux is None:
            raise ValueError(
                "total_flux is required for "
                "normalization_mode='parametric_flux_from_phase1'."
            )
        priors["intensity"] = {"type": "fixed", "value": total_flux}
    elif mode == PIXELIZED:
        priors.pop("intensity", None)
        priors.pop("effective_radius", None)

    if settings is not None:
        phase2_cfg = settings.get("phase2", {})
        if phase2_cfg.get("kinematics_from_truth"):
            from src.pipelines.truth_parameters import priors_fixed_at_truth_except

            free_keys = phase2_cfg.get("free_parameters", ())
            priors = priors_fixed_at_truth_except(
                settings,
                free_keys=free_keys,
                priors_cfg=priors,
            )
            if mode == PARAMETRIC_FLUX_FROM_PHASE1:
                if total_flux is None:
                    raise ValueError(
                        "total_flux is required for "
                        "normalization_mode='parametric_flux_from_phase1'."
                    )
                priors["intensity"] = {"type": "fixed", "value": total_flux}
    return priors
