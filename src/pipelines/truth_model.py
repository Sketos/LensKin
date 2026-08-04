"""Build a truth-parameter model instance without sampling or optimization."""
from __future__ import annotations

from src.model import profiles
from src.pipelines.truth_parameters import truth_parameters_from_settings


def instance_from_debug_vector(settings):
    """
    Build a source instance from ``debug_instance_vector`` using the same Autofit
    model as production fits (``source_model_from_profile``).

    This is the authoritative mapping for vector slot → parameter values,
    including ``centre_0`` / ``centre_1`` order inside the centre tuple.
    """
    import autofit as af

    from src.pipelines.priors import source_model_from_profile

    vector = settings.get("debug_instance_vector")
    if vector is None:
        raise ValueError("settings must include 'debug_instance_vector'")
    priors = settings.get("priors")
    if priors is None:
        raise ValueError("settings must include 'priors' to map debug_instance_vector")

    model_name = settings.get("model_name", "KinMS")
    if model_name == "KinMSPixelized":
        profile_cls = profiles.kinMSPixelized
    elif model_name == "KinMS":
        profile_cls = profiles.kinMS
    elif model_name == "GalPak":
        profile_cls = profiles.GalPaK
    else:
        raise ValueError(f"Unsupported model_name for truth test: {model_name!r}")

    model = af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(profile_cls, priors)
        )
    )
    return model.instance_from_vector(vector=vector)


def truth_instance_from_parameters(params, model_name="KinMS"):
    import autofit as af

    centre = (params["centre_0"], params["centre_1"])
    common = dict(
        centre=centre,
        z_centre=params["z_centre"],
        inclination=params["inclination"],
        phi=params["phi"],
        turnover_radius=params["turnover_radius"],
        maximum_velocity=params["maximum_velocity"],
        velocity_dispersion=params["velocity_dispersion"],
    )

    if model_name == "KinMSPixelized":
        profile = profiles.kinMSPixelized(
            vmax_black_hole=params.get("vmax_black_hole", 0.0),
            **common,
        )
    elif model_name == "KinMS":
        profile = profiles.kinMS(
            intensity=params.get("intensity", 0.05),
            effective_radius=params.get("effective_radius", 0.05),
            vmax_black_hole=params.get("vmax_black_hole", 0.0),
            **common,
        )
    elif model_name == "GalPak":
        profile = profiles.GalPaK(
            intensity=params.get("intensity", 0.05),
            effective_radius=params.get("effective_radius", 0.05),
            **common,
        )
    else:
        raise ValueError(f"Unsupported model_name for truth test: {model_name!r}")

    return af.Collection(galaxies=af.Collection(source=profile))


def truth_instance_from_settings(settings):
    if (
        settings.get("debug_instance_vector") is not None
        and settings.get("priors") is not None
        and "truth_parameters" not in settings
    ):
        return instance_from_debug_vector(settings)

    params = truth_parameters_from_settings(settings)
    return truth_instance_from_parameters(
        params=params,
        model_name=settings.get("model_name", "KinMS"),
    )
