"""Tests for parametric lens-centre optimization wiring."""
import json
from pathlib import Path

from src.model import profiles
from src.pipelines.lens_model import free_lens_centre_from_settings
from src.pipelines.normalization import (
    PARAMETRIC_FLUX_FROM_PHASE1,
    priors_for_normalization_mode,
)
from src.pipelines.runner import parametric_model_from_settings
from src.pipelines.runner_pixelized import _phase2_model_from_settings


def _mock40_settings():
    path = Path(__file__).resolve().parents[1] / "settings/runners/kinms_mock_parametric_mock40.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _flux_settings():
    path = (
        Path(__file__).resolve().parents[1]
        / "settings/runners/kinms_mock_parametric_flux.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_free_lens_centre_adds_two_parameters():
    settings = _mock40_settings()
    assert free_lens_centre_from_settings(settings)

    model_free = parametric_model_from_settings(settings, profiles.kinMS)
    assert model_free.info.count("Total Free Parameters = 12") == 1

    settings_fixed = dict(settings)
    settings_fixed["free_lens_centre"] = False
    model_fixed = parametric_model_from_settings(settings_fixed, profiles.kinMS)
    assert model_fixed.info.count("Total Free Parameters = 10") == 1


def test_lens_centre_vector_order_is_after_source():
    settings = _mock40_settings()
    model = parametric_model_from_settings(settings, profiles.kinMS)
    mass = settings["lens_mass_model"]
    vector = settings["debug_instance_vector"] + [mass["centre_0"], mass["centre_1"]]
    instance = model.instance_from_vector(vector=vector)
    assert tuple(instance.galaxies.lens.mass.centre) == (
        mass["centre_0"],
        mass["centre_1"],
    )


def test_phase2_free_lens_centre_centred_on_phase1():
    settings = _flux_settings()
    assert free_lens_centre_from_settings(settings)
    priors = priors_for_normalization_mode(
        settings["priors"],
        mode=PARAMETRIC_FLUX_FROM_PHASE1,
        settings=settings,
        total_flux=1.0,
    )
    phase1_centre = (0.22, 0.07)
    model = _phase2_model_from_settings(
        settings,
        profiles.kinMS,
        priors,
        phase1_lens_centre=phase1_centre,
    )
    assert hasattr(model.galaxies, "lens")
    half = float(settings["lens_centre_half_width"])
    c0 = model.galaxies.lens.mass.centre_0
    c1 = model.galaxies.lens.mass.centre_1
    assert abs(float(c0.lower_limit) - (phase1_centre[0] - half)) < 1e-9
    assert abs(float(c0.upper_limit) - (phase1_centre[0] + half)) < 1e-9
    assert abs(float(c1.lower_limit) - (phase1_centre[1] - half)) < 1e-9
    assert abs(float(c1.upper_limit) - (phase1_centre[1] + half)) < 1e-9

    settings_fixed = dict(settings)
    settings_fixed["free_lens_centre"] = False
    model_fixed = _phase2_model_from_settings(
        settings_fixed,
        profiles.kinMS,
        priors,
        phase1_lens_centre=phase1_centre,
    )
    assert not hasattr(model_fixed.galaxies, "lens")
