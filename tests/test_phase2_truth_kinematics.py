"""Tests for phase-2 truth-kinematics priors."""
import json
from pathlib import Path

from src.pipelines.normalization import (
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
    priors_for_normalization_mode,
)
from src.pipelines.truth_parameters import priors_fixed_at_truth_except


def _truth_kinematics_settings():
    path = (
        Path(__file__).resolve().parents[1]
        / "settings/runners/kinms_mock_pixelized_truth_kinematics.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_phase2_priors_fix_kinematics_except_centre():
    settings = _truth_kinematics_settings()
    priors = priors_for_normalization_mode(
        settings["priors"],
        PIXELIZED,
        settings=settings,
    )
    assert priors["phi"]["type"] == "fixed"
    assert priors["phi"]["value"] == 90.0
    assert priors["centre_0"]["type"] == "UniformPrior"
    assert priors["centre_1"]["type"] == "UniformPrior"
    assert priors["maximum_velocity"]["value"] == 320.0


def test_priors_fixed_at_truth_except_matches_phase2_block():
    settings = _truth_kinematics_settings()
    free = settings["phase2"]["free_parameters"]
    direct = priors_fixed_at_truth_except(settings, free_keys=free)
    via_norm = priors_for_normalization_mode(
        settings["priors"], PIXELIZED, settings=settings
    )
    assert direct == via_norm


def _parametric_truth_kinematics_settings():
    path = (
        Path(__file__).resolve().parents[1]
        / "settings/runners/kinms_mock_parametric_truth_kinematics.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_parametric_phase2_priors_fix_sb_and_kinematics_except_centre():
    settings = _parametric_truth_kinematics_settings()
    phase1_total_flux = 1.7
    priors = priors_for_normalization_mode(
        settings["priors"],
        PARAMETRIC_FLUX_FROM_PHASE1,
        total_flux=phase1_total_flux,
        settings=settings,
    )
    assert priors["intensity"]["type"] == "fixed"
    assert priors["intensity"]["value"] == phase1_total_flux
    assert priors["effective_radius"]["value"] == 0.1
    assert priors["phi"]["value"] == 90.0
    assert priors["centre_0"]["type"] == "UniformPrior"
    assert priors["centre_1"]["type"] == "UniformPrior"

