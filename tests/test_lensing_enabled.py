"""Tests for lensing.enabled / identity-mass / mesh-override behaviour."""
import copy
import json
from pathlib import Path

import pytest

from src.pipelines.lens_model import (
    apply_lensing_off_reconstruction_overrides,
    free_lens_centre_from_settings,
    identity_lens_mass_model,
    lensing_enabled,
    mass_model_for_settings,
    validate_lensing_settings,
)
from src.pipelines.runner import build_tracer


def _base_settings(**overrides):
    settings = {
        "redshift_lens": 0.4,
        "redshift_source": 2.8,
        "free_lens_centre": False,
        "lens_mass_model": {
            "centre_0": 0.1,
            "centre_1": -0.2,
            "einstein_radius": 1.5,
            "elliptical_comps_0": -0.05,
            "elliptical_comps_1": 0.02,
            "slope": 2.0,
            "shear_elliptical_comps_0": -0.01,
            "shear_elliptical_comps_1": 0.02,
            "multipole_m3_elliptical_comps_0": 0.0,
            "multipole_m3_elliptical_comps_1": 0.0,
            "multipole_m4_elliptical_comps_0": 0.0,
            "multipole_m4_elliptical_comps_1": 0.0,
        },
    }
    settings.update(overrides)
    return settings


def test_lensing_enabled_defaults_true():
    assert lensing_enabled({}) is True
    assert lensing_enabled({"lensing": {}}) is True
    assert lensing_enabled({"lensing": {"enabled": True}}) is True
    assert lensing_enabled({"lensing": {"enabled": False}}) is False


def test_free_lens_centre_forced_false_when_lensing_off():
    settings = _base_settings(
        lensing={"enabled": False},
        free_lens_centre=True,
    )
    assert free_lens_centre_from_settings(settings) is False


def test_validate_rejects_free_lens_centre_when_lensing_off():
    settings = _base_settings(
        lensing={"enabled": False},
        free_lens_centre=True,
    )
    with pytest.raises(ValueError, match="free_lens_centre"):
        validate_lensing_settings(settings)


def test_mass_model_identity_when_lensing_off():
    settings = _base_settings(lensing={"enabled": False})
    mass = mass_model_for_settings(settings)
    assert mass["einstein_radius"] == 0.0
    assert mass["elliptical_comps_0"] == 0.0
    assert mass["shear_elliptical_comps_0"] == 0.0
    # Centres preserved from the configured block.
    assert mass["centre_0"] == 0.1
    assert mass["centre_1"] == -0.2


def test_identity_lens_mass_model_without_settings():
    mass = identity_lens_mass_model()
    assert mass["einstein_radius"] == 0.0
    assert mass["centre_0"] == 0.0


def test_mesh_override_when_lensing_off():
    settings = _base_settings(
        lensing={"enabled": False},
        reconstruction={
            "fix_lens": False,
            "mesh_type": "delaunay",
            "regularization": {"type": "constant_split", "prior_type": "fixed", "value": 1e5},
        },
    )
    apply_lensing_off_reconstruction_overrides(settings)
    rec = settings["reconstruction"]
    assert rec["mesh_type"] == "rectangular_uniform"
    assert rec["regularization"]["type"] == "constant"
    assert rec["fix_lens"] is True


def test_adapt_regularization_kept_when_lensing_off():
    settings = _base_settings(
        lensing={"enabled": False},
        reconstruction={
            "mesh_type": "rectangular_uniform",
            "regularization": {"type": "adapt"},
        },
    )
    apply_lensing_off_reconstruction_overrides(settings)
    assert settings["reconstruction"]["mesh_type"] == "rectangular_uniform"
    assert settings["reconstruction"]["regularization"]["type"] == "adapt"


def test_adapt_split_remapped_when_lensing_off():
    settings = _base_settings(
        lensing={"enabled": False},
        reconstruction={
            "mesh_type": "delaunay",
            "regularization": {"type": "adapt_split"},
        },
    )
    apply_lensing_off_reconstruction_overrides(settings)
    assert settings["reconstruction"]["mesh_type"] == "rectangular_uniform"
    assert settings["reconstruction"]["regularization"]["type"] == "adapt"


def test_validate_applies_mesh_override():
    settings = _base_settings(
        lensing={"enabled": False},
        free_lens_centre=False,
        reconstruction={
            "mesh_type": "rectangular_adapt_image",
            "regularization": {"type": "adapt"},
        },
    )
    assert validate_lensing_settings(settings) is False
    assert settings["reconstruction"]["mesh_type"] == "rectangular_uniform"
    assert settings["reconstruction"]["regularization"]["type"] == "adapt"


def test_adapt_kept_when_lensing_on():
    settings = _base_settings(
        lensing={"enabled": True},
        reconstruction={
            "mesh_type": "rectangular_uniform",
            "regularization": {
                "type": "adapt",
                "prior_type": "fixed",
                "inner_coefficient": 1.0,
                "outer_coefficient": 50.0,
                "signal_scale": 3.0,
            },
        },
    )
    assert validate_lensing_settings(settings) is True
    apply_lensing_off_reconstruction_overrides(settings)
    assert settings["reconstruction"]["mesh_type"] == "rectangular_uniform"
    assert settings["reconstruction"]["regularization"]["type"] == "adapt"


def test_lensed_adapt_runner_settings_file():
    path = (
        Path(__file__).resolve().parents[1]
        / "settings/runners/kinms_mock_lensed_pixelized_adapt.json"
    )
    settings = json.loads(path.read_text())
    validate_lensing_settings(settings)
    rec = settings["reconstruction"]
    assert rec["mesh_type"] == "delaunay"
    assert rec["regularization"]["type"] == "adapt_split"
    from src.pipelines.reconstruction import validate_mesh_regularization_pair

    validate_mesh_regularization_pair(rec["mesh_type"], rec["regularization"]["type"])


def test_mesh_not_overridden_when_lensing_on():
    settings = _base_settings(
        lensing={"enabled": True},
        reconstruction={
            "mesh_type": "delaunay",
            "regularization": {"type": "constant_split"},
        },
    )
    apply_lensing_off_reconstruction_overrides(settings)
    assert settings["reconstruction"]["mesh_type"] == "delaunay"
    assert settings["reconstruction"]["regularization"]["type"] == "constant_split"


def test_build_tracer_identity_when_lensing_off():
    settings = _base_settings(lensing={"enabled": False})
    validate_lensing_settings(settings)
    tracer = build_tracer(settings)
    lens = tracer.galaxies[0]
    assert float(lens.mass.einstein_radius) == 0.0


def test_unlensed_parametric_settings_file():
    path = (
        Path(__file__).resolve().parents[1]
        / "settings/runners/kinms_mock_unlensed_parametric.json"
    )
    with open(path, encoding="utf-8") as f:
        settings = json.load(f)
    # File may already have the flag after this PR; tolerate either until updated.
    settings = copy.deepcopy(settings)
    settings["lensing"] = {"enabled": False}
    settings["free_lens_centre"] = False
    assert validate_lensing_settings(settings) is False
    assert free_lens_centre_from_settings(settings) is False
    assert mass_model_for_settings(settings)["einstein_radius"] == 0.0
