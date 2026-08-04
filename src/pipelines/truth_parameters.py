"""Truth-parameter name order and parsing (no autofit dependency)."""

PIXELIZED_KINEMATIC_KEYS = (
    "z_centre",
    "centre_0",
    "centre_1",
    "phi",
    "inclination",
    "turnover_radius",
    "maximum_velocity",
    "velocity_dispersion",
    "vmax_black_hole",
)

PARAMETRIC_KINEMATIC_KEYS = (
    "z_centre",
    "intensity",
    "effective_radius",
    "phi",
    "inclination",
    "maximum_velocity",
    "velocity_dispersion",
    "turnover_radius",
    "centre_0",
    "centre_1",
    "vmax_black_hole",
)

PARAMETRIC_KINMS_DEBUG_ORDER = [
    "z_centre",
    "intensity",
    "effective_radius",
    "phi",
    "inclination",
    "maximum_velocity",
    "velocity_dispersion",
    "turnover_radius",
    # Autofit ``model.info``: centre_0 is index 8, centre_1 is index 9.
    "centre_0",
    "centre_1",
    "vmax_black_hole",
]

PIXELIZED_KINMS_DEBUG_ORDER = [
    "z_centre",
    "centre_0",
    "centre_1",
    "phi",
    "inclination",
    "turnover_radius",
    "maximum_velocity",
    "velocity_dispersion",
    "vmax_black_hole",
]


def truth_parameters_from_settings(settings):
    if "truth_parameters" in settings:
        return dict(settings["truth_parameters"])

    vector = settings.get("debug_instance_vector")
    if vector is None:
        raise ValueError(
            "Settings must include 'truth_parameters' or 'debug_instance_vector'."
        )

    model_name = settings.get("model_name", "KinMS")
    if model_name == "KinMSPixelized":
        keys = PIXELIZED_KINMS_DEBUG_ORDER
    else:
        keys = PARAMETRIC_KINMS_DEBUG_ORDER

    params = dict(zip(keys[: len(vector)], vector))
    if "vmax_black_hole" not in params:
        params["vmax_black_hole"] = 0.0

    if (
        settings.get("priors") is not None
        and "truth_parameters" not in settings
        and model_name in ("KinMS", "KinMSPixelized")
    ):
        from src.pipelines.truth_model import instance_from_debug_vector

        centre = instance_from_debug_vector(settings).galaxies.source.centre
        params["centre_0"] = float(centre[0])
        params["centre_1"] = float(centre[1])

    return params


def kinematic_parameter_keys_for_settings(settings):
    model_name = settings.get("model_name", "KinMS")
    if model_name == "KinMSPixelized":
        return PIXELIZED_KINEMATIC_KEYS
    return PARAMETRIC_KINEMATIC_KEYS


def priors_fixed_at_truth_except(settings, free_keys=(), priors_cfg=None):
    """
    Build phase-2 priors with source parameters fixed at ``truth_parameters``
    (or ``debug_instance_vector``), leaving only ``free_keys`` as sampled priors.
    """
    truth = truth_parameters_from_settings(settings)
    base = dict(priors_cfg or settings.get("priors", {}))
    free_keys = set(free_keys)
    out = {}
    for key in kinematic_parameter_keys_for_settings(settings):
        if key in free_keys:
            if key not in base:
                raise ValueError(
                    f"free parameter {key!r} requires a prior entry in settings['priors']"
                )
            out[key] = base[key]
        else:
            if key not in truth:
                raise ValueError(
                    f"truth value for {key!r} missing; add truth_parameters or "
                    "debug_instance_vector to settings."
                )
            out[key] = {"type": "fixed", "value": truth[key]}
    return out
