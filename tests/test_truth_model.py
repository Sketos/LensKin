"""
Unit tests for truth-parameter helpers (no autolens / KinMS required).
"""
from src.pipelines.truth_parameters import (
    PARAMETRIC_KINMS_DEBUG_ORDER,
    truth_parameters_from_settings,
)


def test_truth_parameters_from_debug_vector():
    settings = {
        "model_name": "KinMS",
        "debug_instance_vector": [
            0.0, 0.05, 0.05, 90.0, 65.0, 320.0, 35.0, 0.05, 0.0, 0.0
        ],
    }
    params = truth_parameters_from_settings(settings)
    assert params["z_centre"] == 0.0
    assert params["intensity"] == 0.05
    assert params["maximum_velocity"] == 320.0
    assert params["centre_0"] == 0.0
    assert params["centre_1"] == 0.0
    assert params["vmax_black_hole"] == 0.0


def test_truth_parameters_centre_vector_order_matches_autofit():
    """Autofit vector slots 8/9 are centre_0 / centre_1 (see model.info)."""
    settings = {
        "model_name": "KinMS",
        "debug_instance_vector": [
            0.0, 0.05, 0.05, 90.0, 65.0, 320.0, 35.0, 0.05, -0.22, 0.11
        ],
    }
    params = truth_parameters_from_settings(settings)
    assert params["centre_0"] == -0.22
    assert params["centre_1"] == 0.11


def test_truth_parameters_explicit_block():
    settings = {
        "truth_parameters": {
            "z_centre": 1.0,
            "intensity": 0.1,
            "effective_radius": 0.04,
            "phi": 80.0,
            "inclination": 60.0,
            "maximum_velocity": 300.0,
            "velocity_dispersion": 30.0,
            "turnover_radius": 0.04,
            "centre_0": 0.01,
            "centre_1": -0.02,
        }
    }
    params = truth_parameters_from_settings(settings)
    assert params["z_centre"] == 1.0
    assert params["centre_1"] == -0.02
