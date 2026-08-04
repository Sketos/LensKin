"""GalPaK support for parametric and parametric_flux_from_phase1 modes."""

import pytest

from src.pipelines.normalization import (
    PARAMETRIC,
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
    intensity_from_phase1_intflux,
    priors_for_normalization_mode,
    validate_normalization_settings,
)


def test_galpak_parametric_allowed():
    mode = validate_normalization_settings(
        {
            "model_name": "GalPak",
            "normalization_mode": PARAMETRIC,
        }
    )
    assert mode == PARAMETRIC


def test_galpak_parametric_flux_requires_reconstruction():
    with pytest.raises(ValueError, match="reconstruction"):
        validate_normalization_settings(
            {
                "model_name": "GalPak",
                "normalization_mode": PARAMETRIC_FLUX_FROM_PHASE1,
            }
        )


def test_galpak_parametric_flux_allowed_with_reconstruction():
    mode = validate_normalization_settings(
        {
            "model_name": "GalPak",
            "normalization_mode": PARAMETRIC_FLUX_FROM_PHASE1,
            "reconstruction": {"mesh_type": "rectangular_uniform"},
        }
    )
    assert mode == PARAMETRIC_FLUX_FROM_PHASE1


def test_galpak_pixelized_still_blocked():
    with pytest.raises(ValueError, match="pixelized"):
        validate_normalization_settings(
            {
                "model_name": "GalPak",
                "normalization_mode": PIXELIZED,
                "reconstruction": {"mesh_type": "rectangular_uniform"},
            }
        )


def test_intensity_from_phase1_intflux_kinms_passthrough():
    assert intensity_from_phase1_intflux("KinMS", 12.5, 30.0) == 12.5


def test_intensity_from_phase1_intflux_galpak_divides_by_dv():
    assert intensity_from_phase1_intflux("GalPak", 12.5, 30.0) == pytest.approx(12.5 / 30.0)


def test_priors_fix_intensity_for_galpak_mode2():
    priors = priors_for_normalization_mode(
        priors_cfg={"effective_radius": {"type": "fixed", "value": 0.1}},
        mode=PARAMETRIC_FLUX_FROM_PHASE1,
        total_flux=0.4,
    )
    assert priors["intensity"] == {"type": "fixed", "value": 0.4}
