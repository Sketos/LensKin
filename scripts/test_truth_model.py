#!/usr/bin/env python3
"""
Evaluate the truth model directly against data — no Nautilus or optimizer.

Confirms that KinMS cube generation, lensing, and visibility prediction are
wired correctly at known truth parameters.

Examples:
  python scripts/test_truth_model.py \\
    --settings settings/runners/kinms_mock_parametric_bbox512.json

  python scripts/test_truth_model.py \\
    --settings settings/runners/kinms_mock_parametric_flux.json \\
    --run-phase1 --lens-centre phase1

  python scripts/test_truth_model.py \\
    --settings settings/runners/kinms_mock_parametric_mock40.json \\
    --optimize-intensity-and-re

  python scripts/test_truth_model.py \\
    --settings settings/runners/kinms_mock_parametric_mock40.json \\
    --optimize-lens-centre

Plots are written under ``<output_path>/truth_model_test/`` by default.
"""
import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup

setup(__file__)

import numpy as np
from scipy import optimize as scipy_optimize

from src.analysis import analysis as analysis_mod
from src.analysis import visualizer as visualizer_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines import reconstruction
from src.pipelines.normalization import PIXELIZED, requires_phase1, validate_normalization_settings
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.truth_parameters import (
    PARAMETRIC_KINMS_DEBUG_ORDER,
    truth_parameters_from_settings,
)
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils


def _finite_stats(arr, label):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    frac = float(finite.mean()) if arr.size else 0.0
    print(f"  {label}")
    print(f"    shape = {arr.shape}")
    print(f"    finite fraction = {frac:.6f}")
    if finite.any():
        vals = arr[finite]
        print(f"    min / max = {vals.min():.4g} / {vals.max():.4g}")
        print(f"    sum = {vals.sum():.4g}")
    else:
        print("    min / max = nan / nan")


def _visibility_fit_stats(analysis_instance, model_data):
    fit = analysis_mod.fit.DatasetFit(
        masked_dataset=analysis_instance.masked_dataset,
        model_data=model_data,
    )
    mask = fit.mask == False
    residual = fit.data[mask] - model_data[mask]
    sigma = fit.noise_map[mask]
    chi2 = float(fit.chi_squared)
    n_data = int(mask.sum())
    rms = float(np.sqrt(np.mean(residual**2)))
    median_sigma = float(np.median(sigma))
    normalized_rms = float(
        np.sqrt(np.mean((residual / sigma) ** 2))
    )
    return {
        "log_likelihood": float(fit.likelihood),
        "chi_squared": chi2,
        "chi_squared_per_datum": chi2 / n_data if n_data else float("nan"),
        "n_data": n_data,
        "residual_rms": rms,
        "median_sigma": median_sigma,
        "normalized_residual_rms": normalized_rms,
        "reduced_chi2": chi2 / n_data if n_data else float("nan"),
    }


def _intensity_bounds_from_settings(settings, lower=None, upper=None):
    cfg = settings.get("priors", {}).get("intensity", {})
    if lower is None or upper is None:
        if cfg.get("type") == "LogUniformPrior":
            lo = float(cfg["lower_limit"])
            hi = float(cfg["upper_limit"])
        elif cfg.get("type") == "UniformPrior":
            lo = float(cfg["lower_limit"])
            hi = float(cfg["upper_limit"])
        elif cfg.get("type") == "fixed":
            val = float(cfg["value"])
            lo, hi = val, val
        else:
            lo, hi = 0.01, 0.15
    else:
        lo = float(lower)
        hi = float(upper)
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Invalid intensity bounds: ({lo}, {hi})")
    return lo, hi


def _effective_radius_bounds_from_settings(settings, lower=None, upper=None):
    cfg = settings.get("priors", {}).get("effective_radius", {})
    if lower is None or upper is None:
        if cfg.get("type") == "LogUniformPrior":
            lo = float(cfg["lower_limit"])
            hi = float(cfg["upper_limit"])
        elif cfg.get("type") == "UniformPrior":
            lo = float(cfg["lower_limit"])
            hi = float(cfg["upper_limit"])
        elif cfg.get("type") == "fixed":
            val = float(cfg["value"])
            lo, hi = val, val
        else:
            lo, hi = 0.03, 0.15
    else:
        lo = float(lower)
        hi = float(upper)
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Invalid effective_radius bounds: ({lo}, {hi})")
    return lo, hi


def _optimize_intensity_at_truth(
    settings,
    analysis_instance,
    truth_intensity,
    intensity_lower=None,
    intensity_upper=None,
):
    """
    Hold all non-flux truth parameters fixed and maximize log-likelihood in
    ``intensity`` (KinMS ``intFlux``) only.
    """
    lo, hi = _intensity_bounds_from_settings(
        settings,
        lower=intensity_lower,
        upper=intensity_upper,
    )

    def objective(log10_intensity):
        intensity = float(10.0 ** log10_intensity)
        instance = truth_instance_from_settings(settings)
        instance.galaxies.source.intensity = intensity
        try:
            return -float(
                analysis_instance.log_likelihood_function(instance=instance)
            )
        except Exception:
            return 1.0e30

    log_lo, log_hi = np.log10(lo), np.log10(hi)
    result = scipy_optimize.minimize_scalar(
        objective,
        bounds=(log_lo, log_hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    best_intensity = float(10.0 ** result.x)
    best_instance = truth_instance_from_settings(settings)
    best_instance.galaxies.source.intensity = best_intensity
    best_model_data = analysis_instance.model_data_from_instance(
        instance=best_instance
    )
    best_stats = _visibility_fit_stats(analysis_instance, best_model_data)

    truth_instance = truth_instance_from_settings(settings)
    truth_instance.galaxies.source.intensity = float(truth_intensity)
    truth_model_data = analysis_instance.model_data_from_instance(
        instance=truth_instance
    )
    truth_stats = _visibility_fit_stats(analysis_instance, truth_model_data)

    return {
        "bounds": (lo, hi),
        "truth_intensity": float(truth_intensity),
        "best_intensity": best_intensity,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "truth_stats": truth_stats,
        "best_stats": best_stats,
        "best_instance": best_instance,
        "best_model_data": best_model_data,
    }


def _print_intensity_optimization_report(report):
    lo, hi = report["bounds"]
    truth = report["truth_stats"]
    best = report["best_stats"]
    print("\n=== Intensity optimization (other parameters fixed at truth) ===\n")
    print(f"  Search range (intensity / intFlux) = [{lo:.4g}, {hi:.4g}] Jy km/s")
    print(f"  Truth intensity                  = {report['truth_intensity']:.6g} Jy km/s")
    print(f"  Best-fit intensity               = {report['best_intensity']:.6g} Jy km/s")
    if report["truth_intensity"] > 0:
        print(
            f"  Best / truth intensity ratio     = "
            f"{report['best_intensity'] / report['truth_intensity']:.4g}"
        )
    print(f"  Optimizer success                = {report['optimizer_success']}")
    print(f"  Optimizer evaluations            = {report['optimizer_nfev']}")
    print("\n  At truth intensity:")
    print(f"    chi-squared / N = {truth['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {truth['log_likelihood']:.2f}")
    print("\n  At best-fit intensity:")
    print(f"    chi-squared / N = {best['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {best['log_likelihood']:.2f}")
    delta = best["log_likelihood"] - truth["log_likelihood"]
    print(f"\n  Δ log likelihood (best − truth)  = {delta:+.2f}")
    if best["chi_squared_per_datum"] < truth["chi_squared_per_datum"] - 1e-6:
        print(
            "  INTENSITY GAIN — a different intFlux improves the UV fit; "
            "simobserve inbright / flux scaling may explain the offset."
        )
    else:
        print(
            "  No improvement — truth intensity is already optimal within "
            "the search bounds."
        )


def _optimize_intensity_and_effective_radius_at_truth(
    settings,
    analysis_instance,
    truth_intensity,
    truth_effective_radius,
    intensity_lower=None,
    intensity_upper=None,
    effective_radius_lower=None,
    effective_radius_upper=None,
    start_from="prior_center",
):
    """
    Hold all parameters except ``intensity`` and ``effective_radius`` fixed at
    truth and maximize log-likelihood in both.
    """
    i_lo, i_hi = _intensity_bounds_from_settings(
        settings,
        lower=intensity_lower,
        upper=intensity_upper,
    )
    r_lo, r_hi = _effective_radius_bounds_from_settings(
        settings,
        lower=effective_radius_lower,
        upper=effective_radius_upper,
    )

    def neg_log_likelihood(log10_intensity, log10_effective_radius):
        instance = truth_instance_from_settings(settings)
        instance.galaxies.source.intensity = float(10.0 ** log10_intensity)
        instance.galaxies.source.effective_radius = float(
            10.0 ** log10_effective_radius
        )
        try:
            return -float(
                analysis_instance.log_likelihood_function(instance=instance)
            )
        except Exception:
            return 1.0e30

    def objective(params):
        return neg_log_likelihood(params[0], params[1])

    if start_from == "truth":
        x0 = [np.log10(truth_intensity), np.log10(truth_effective_radius)]
    elif start_from == "prior_center":
        x0 = [
            0.5 * (np.log10(i_lo) + np.log10(i_hi)),
            0.5 * (np.log10(r_lo) + np.log10(r_hi)),
        ]
    else:
        raise ValueError(
            f"Unsupported start_from={start_from!r}; use 'truth' or 'prior_center'."
        )

    bounds = [
        (np.log10(i_lo), np.log10(i_hi)),
        (np.log10(r_lo), np.log10(r_hi)),
    ]
    result = scipy_optimize.minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
    )
    best_intensity = float(10.0 ** result.x[0])
    best_effective_radius = float(10.0 ** result.x[1])

    best_instance = truth_instance_from_settings(settings)
    best_instance.galaxies.source.intensity = best_intensity
    best_instance.galaxies.source.effective_radius = best_effective_radius
    best_model_data = analysis_instance.model_data_from_instance(
        instance=best_instance
    )
    best_stats = _visibility_fit_stats(analysis_instance, best_model_data)

    truth_instance = truth_instance_from_settings(settings)
    truth_instance.galaxies.source.intensity = float(truth_intensity)
    truth_instance.galaxies.source.effective_radius = float(truth_effective_radius)
    truth_model_data = analysis_instance.model_data_from_instance(
        instance=truth_instance
    )
    truth_stats = _visibility_fit_stats(analysis_instance, truth_model_data)

    start_instance = truth_instance_from_settings(settings)
    start_instance.galaxies.source.intensity = float(10.0 ** x0[0])
    start_instance.galaxies.source.effective_radius = float(10.0 ** x0[1])
    start_model_data = analysis_instance.model_data_from_instance(
        instance=start_instance
    )
    start_stats = _visibility_fit_stats(analysis_instance, start_model_data)

    return {
        "intensity_bounds": (i_lo, i_hi),
        "effective_radius_bounds": (r_lo, r_hi),
        "start_from": start_from,
        "start_intensity": float(10.0 ** x0[0]),
        "start_effective_radius": float(10.0 ** x0[1]),
        "truth_intensity": float(truth_intensity),
        "truth_effective_radius": float(truth_effective_radius),
        "best_intensity": best_intensity,
        "best_effective_radius": best_effective_radius,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_message": str(result.message),
        "start_stats": start_stats,
        "truth_stats": truth_stats,
        "best_stats": best_stats,
        "best_instance": best_instance,
        "best_model_data": best_model_data,
    }


def _print_intensity_and_effective_radius_optimization_report(report):
    i_lo, i_hi = report["intensity_bounds"]
    r_lo, r_hi = report["effective_radius_bounds"]
    truth = report["truth_stats"]
    best = report["best_stats"]
    start = report["start_stats"]

    print(
        "\n=== Intensity + effective_radius optimization "
        "(other parameters fixed at truth) ===\n"
    )
    print(f"  Optimizer start          = {report['start_from']}")
    print(f"  Start intensity          = {report['start_intensity']:.6g} Jy km/s")
    print(
        f"  Start effective_radius   = {report['start_effective_radius']:.6g} arcsec"
    )
    print(f"  Search range (intensity) = [{i_lo:.4g}, {i_hi:.4g}] Jy km/s")
    print(f"  Search range (re)        = [{r_lo:.4g}, {r_hi:.4g}] arcsec")
    print(f"  Truth intensity          = {report['truth_intensity']:.6g} Jy km/s")
    print(
        f"  Truth effective_radius   = {report['truth_effective_radius']:.6g} arcsec"
    )
    print(f"  Best-fit intensity       = {report['best_intensity']:.6g} Jy km/s")
    print(
        f"  Best-fit effective_radius = {report['best_effective_radius']:.6g} arcsec"
    )
    if report["truth_intensity"] > 0:
        print(
            f"  Best / truth intensity ratio = "
            f"{report['best_intensity'] / report['truth_intensity']:.4g}"
        )
    if report["truth_effective_radius"] > 0:
        print(
            f"  Best / truth re ratio        = "
            f"{report['best_effective_radius'] / report['truth_effective_radius']:.4g}"
        )
    print(f"  Optimizer success        = {report['optimizer_success']}")
    print(f"  Optimizer evaluations    = {report['optimizer_nfev']}")
    if report.get("optimizer_message"):
        print(f"  Optimizer message        = {report['optimizer_message']}")

    print("\n  At optimizer start:")
    print(f"    chi-squared / N = {start['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {start['log_likelihood']:.2f}")
    print("\n  At truth (input) values:")
    print(f"    chi-squared / N = {truth['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {truth['log_likelihood']:.2f}")
    print("\n  At best-fit values:")
    print(f"    chi-squared / N = {best['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {best['log_likelihood']:.2f}")
    delta = best["log_likelihood"] - truth["log_likelihood"]
    print(f"\n  Δ log likelihood (best − truth) = {delta:+.2f}")

    bound_margin = 0.02
    if report["best_intensity"] <= i_lo * (1.0 + bound_margin):
        print(
            "  INTENSITY AT LOWER PRIOR — best-fit intensity sits on the "
            "search lower bound."
        )
    if report["best_intensity"] >= i_hi * (1.0 - bound_margin):
        print(
            "  INTENSITY AT UPPER PRIOR — best-fit intensity sits on the "
            "search upper bound."
        )
    if report["best_effective_radius"] <= r_lo * (1.0 + bound_margin):
        print(
            "  RE AT LOWER PRIOR — best-fit effective_radius sits on the "
            "search lower bound."
        )
    if report["best_effective_radius"] >= r_hi * (1.0 - bound_margin):
        print(
            "  RE AT UPPER PRIOR — best-fit effective_radius sits on the "
            "search upper bound."
        )

    i_close = (
        abs(report["best_intensity"] - report["truth_intensity"])
        <= 0.1 * report["truth_intensity"]
    )
    re_close = (
        abs(report["best_effective_radius"] - report["truth_effective_radius"])
        <= 0.1 * report["truth_effective_radius"]
    )
    if i_close and re_close:
        print(
            "  RECOVERY PASS — best-fit intensity and effective_radius are "
            "within 10% of the input truth values."
        )
    else:
        if not i_close:
            print(
                "  INTENSITY OFFSET — best-fit intFlux differs from input truth "
                "by more than 10%; simobserve flux renorm may be required."
            )
        if not re_close:
            print(
                "  RE OFFSET — best-fit effective_radius differs from input "
                "truth by more than 10%; check source_grid / mock generation "
                "alignment."
            )


def _lens_centre_bounds_from_settings(
    settings,
    half_width=0.05,
    centre_0_lower=None,
    centre_0_upper=None,
    centre_1_lower=None,
    centre_1_upper=None,
):
    mass = settings["lens_mass_model"]
    c0 = float(mass["centre_0"])
    c1 = float(mass["centre_1"])
    lo0 = float(centre_0_lower) if centre_0_lower is not None else c0 - half_width
    hi0 = float(centre_0_upper) if centre_0_upper is not None else c0 + half_width
    lo1 = float(centre_1_lower) if centre_1_lower is not None else c1 - half_width
    hi1 = float(centre_1_upper) if centre_1_upper is not None else c1 + half_width
    if hi0 <= lo0 or hi1 <= lo1:
        raise ValueError(
            f"Invalid lens-centre bounds: centre_0=({lo0}, {hi0}), "
            f"centre_1=({lo1}, {hi1})"
        )
    return (lo0, hi0), (lo1, hi1)


def _analysis_with_lens_centre(pipeline, settings, centre):
    analysis = pipeline["analysis"]
    tracer = build_tracer(settings, centre=centre)
    return analysis_mod.Analysis(
        masked_dataset=analysis.masked_dataset,
        transformers=analysis.transformers,
        tracer=tracer,
        settings=settings,
    )


def _mom0_peak_yx(cube, grid_2d, z_step_kms):
    mom0 = np.asarray(cube).sum(axis=0) * z_step_kms
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    i, j = np.unravel_index(np.nanargmax(mom0), mom0.shape)
    return float(coords[i, j, 0]), float(coords[i, j, 1])


def _optimize_lens_centre_at_truth(
    settings,
    pipeline,
    truth_lens_centre,
    half_width=0.05,
    centre_0_lower=None,
    centre_0_upper=None,
    centre_1_lower=None,
    centre_1_upper=None,
):
    """
    Hold source parameters at truth and maximize log-likelihood in lens mass
    centre (centre_0, centre_1) only.
    """
    from src.utils import autolens_utils

    (c0_lo, c0_hi), (c1_lo, c1_hi) = _lens_centre_bounds_from_settings(
        settings,
        half_width=half_width,
        centre_0_lower=centre_0_lower,
        centre_0_upper=centre_0_upper,
        centre_1_lower=centre_1_lower,
        centre_1_upper=centre_1_upper,
    )
    instance = truth_instance_from_settings(settings)
    image_grid = pipeline["image_grid_3d"].grid_2d
    z_step = pipeline["z_step_kms"]

    def neg_log_likelihood(params):
        centre = (float(params[0]), float(params[1]))
        analysis = _analysis_with_lens_centre(pipeline, settings, centre)
        try:
            return -float(analysis.log_likelihood_function(instance=instance))
        except Exception:
            return 1.0e30

    x0 = [float(truth_lens_centre[0]), float(truth_lens_centre[1])]
    bounds = [(c0_lo, c0_hi), (c1_lo, c1_hi)]
    result = scipy_optimize.minimize(
        neg_log_likelihood,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
    )
    best_centre = (float(result.x[0]), float(result.x[1]))
    best_analysis = _analysis_with_lens_centre(pipeline, settings, best_centre)
    truth_analysis = _analysis_with_lens_centre(
        pipeline, settings, tuple(truth_lens_centre)
    )

    best_model_data = best_analysis.model_data_from_instance(instance=instance)
    truth_model_data = truth_analysis.model_data_from_instance(instance=instance)
    best_stats = _visibility_fit_stats(best_analysis, best_model_data)
    truth_stats = _visibility_fit_stats(truth_analysis, truth_model_data)

    best_source_cube = best_analysis.model_cube_from_instance(instance=instance)
    from src.utils import analysis_utils

    best_lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=best_source_cube,
        tracer=best_analysis.tracer,
        grid=best_analysis.masked_dataset.grid_3d.grid_2d,
        z_mask=best_analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
        output_shape=best_analysis.masked_dataset.grid_3d.shape_2d,
    )
    truth_source_cube = truth_analysis.model_cube_from_instance(instance=instance)
    truth_lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=truth_source_cube,
        tracer=truth_analysis.tracer,
        grid=truth_analysis.masked_dataset.grid_3d.grid_2d,
        z_mask=truth_analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
        output_shape=truth_analysis.masked_dataset.grid_3d.shape_2d,
    )

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=best_analysis.masked_dataset.data,
        transformers=best_analysis.transformers,
    )
    data_peak = _mom0_peak_yx(dirty_data, image_grid, z_step)
    truth_peak = _mom0_peak_yx(truth_lensed_cube, image_grid, z_step)
    best_peak = _mom0_peak_yx(best_lensed_cube, image_grid, z_step)

    return {
        "bounds": {"centre_0": (c0_lo, c0_hi), "centre_1": (c1_lo, c1_hi)},
        "truth_lens_centre": tuple(truth_lens_centre),
        "best_lens_centre": best_centre,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_message": str(result.message),
        "truth_stats": truth_stats,
        "best_stats": best_stats,
        "best_analysis": best_analysis,
        "best_instance": instance,
        "best_model_data": best_model_data,
        "best_source_cube": best_source_cube,
        "best_lensed_cube": best_lensed_cube,
        "truth_peak": truth_peak,
        "best_peak": best_peak,
        "data_peak": data_peak,
        "truth_peak_offset": (
            truth_peak[0] - data_peak[0],
            truth_peak[1] - data_peak[1],
        ),
        "best_peak_offset": (
            best_peak[0] - data_peak[0],
            best_peak[1] - data_peak[1],
        ),
    }


def _print_lens_centre_optimization_report(report):
    (c0_lo, c0_hi) = report["bounds"]["centre_0"]
    (c1_lo, c1_hi) = report["bounds"]["centre_1"]
    truth = report["truth_stats"]
    best = report["best_stats"]
    tc = report["truth_lens_centre"]
    bc = report["best_lens_centre"]

    print(
        "\n=== Lens-centre optimization "
        "(source parameters fixed at truth) ===\n"
    )
    print(f"  Search range centre_0 (x) = [{c0_lo:.4f}, {c0_hi:.4f}] arcsec")
    print(f"  Search range centre_1 (y) = [{c1_lo:.4f}, {c1_hi:.4f}] arcsec")
    print(f"  Truth lens centre         = ({tc[0]:.4f}, {tc[1]:.4f})")
    print(f"  Best-fit lens centre      = ({bc[0]:.4f}, {bc[1]:.4f})")
    print(
        f"  Δ lens centre (best − truth) = "
        f"({bc[0] - tc[0]:+.4f}, {bc[1] - tc[1]:+.4f}) arcsec"
    )
    print(f"  Optimizer success         = {report['optimizer_success']}")
    print(f"  Optimizer evaluations     = {report['optimizer_nfev']}")
    if report.get("optimizer_message"):
        print(f"  Optimizer message         = {report['optimizer_message']}")

    print("\n  At truth lens centre:")
    print(f"    chi-squared / N = {truth['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {truth['log_likelihood']:.2f}")
    print("\n  At best-fit lens centre:")
    print(f"    chi-squared / N = {best['chi_squared_per_datum']:.6f}")
    print(f"    log likelihood  = {best['log_likelihood']:.2f}")
    delta = best["log_likelihood"] - truth["log_likelihood"]
    print(f"\n  Δ log likelihood (best − truth) = {delta:+.2f}")

    dp = report["data_peak"]
    tp = report["truth_peak"]
    bp = report["best_peak"]
    toff = report["truth_peak_offset"]
    boff = report["best_peak_offset"]
    print("\n  Mom0 peak positions (y, x):")
    print(f"    dirty data           = ({dp[0]:+.4f}, {dp[1]:+.4f})")
    print(f"    lensed model (truth) = ({tp[0]:+.4f}, {tp[1]:+.4f})")
    print(f"    lensed model (best)  = ({bp[0]:+.4f}, {bp[1]:+.4f})")
    print(
        f"    model − data offset (truth) = ({toff[0]:+.4f}, {toff[1]:+.4f}) arcsec"
    )
    print(
        f"    model − data offset (best)  = ({boff[0]:+.4f}, {boff[1]:+.4f}) arcsec"
    )
    truth_dist = float(np.hypot(toff[0], toff[1]))
    best_dist = float(np.hypot(boff[0], boff[1]))
    print(f"    peak distance (truth) = {truth_dist:.4f} arcsec")
    print(f"    peak distance (best)  = {best_dist:.4f} arcsec")

    if best["chi_squared_per_datum"] < truth["chi_squared_per_datum"] - 1e-6:
        print(
            "  LENS CENTRE GAIN — shifting the lens centre improves the UV fit."
        )
    else:
        print(
            "  No improvement — truth lens centre is already optimal within "
            "the search bounds."
        )


def _print_coordinate_audit(
    settings,
    pipeline,
    source_cube,
    lensed_cube,
    analysis_instance,
    truth_params,
):
    """Report grid extents, centre conventions, and bright-feature positions."""
    from src.utils import autolens_utils

    kinms_grid = pipeline["kinms_grid_3d"]
    image_grid = pipeline["image_grid_3d"]
    z_step = pipeline["z_step_kms"]

    def grid_summary(label, grid_2d):
        coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
        y = coords[:, 0, 0]
        x = coords[0, :, 1]
        dy = abs(float(y[1] - y[0])) if len(y) > 1 else float("nan")
        dx = abs(float(x[1] - x[0])) if len(x) > 1 else float("nan")
        print(
            f"  {label}: {grid_2d.shape_native[0]}², "
            f"y=[{y.min():.4f}, {y.max():.4f}], x=[{x.min():.4f}, {x.max():.4f}], "
            f"pixel=({dy:.4g}, {dx:.4g}) arcsec"
        )

    def peak_yx(cube_2d, grid_2d):
        coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
        i, j = np.unravel_index(np.nanargmax(cube_2d), cube_2d.shape)
        return float(coords[i, j, 0]), float(coords[i, j, 1])

    print("\n=== Coordinate / grid audit ===\n")
    print("  Autolens grids use (y, x) arcsec with y decreasing along rows.")
    print(
        "  Settings lens_mass_model centre_0/centre_1 are passed to "
        "autolens as mass.centre = (centre_0, centre_1)."
    )
    print(
        "  phaseCent = [centre_0, centre_1] = (x, y) arcsec; profile centre tuple is "
        "(centre_0, centre_1) = (x, y)."
    )
    print(
        "  KinMS posAng (= profile phi): PA=0° => major axis along +y; "
        "passed through without offset."
    )
    print(
        "  debug_instance_vector centre slots: index 8 = centre_0 (x), "
        "index 9 = centre_1 (y); see model.info."
    )
    grid_summary("KinMS source grid", kinms_grid.grid_2d)
    grid_summary("Image-plane grid", image_grid.grid_2d)

    inst = analysis_instance.masked_dataset.instance
    if inst is not None and getattr(inst, "obj", None) is not None:
        print(
            f"  KinMS field: xs={inst.obj.xs:.4g}″, ys={inst.obj.ys:.4g}″, "
            f"cellSize={inst.obj.cellSize:.6g}″"
        )

    lc = pipeline["lens_centre"]
    print(f"  Lens centre (tracer): ({lc[0]:.4f}, {lc[1]:.4f})")
    print(
        f"  Truth source centre: ({truth_params['centre_0']:.4f}, "
        f"{truth_params['centre_1']:.4f})"
    )

    src_mom0 = source_cube.sum(axis=0) * z_step
    sy, sx = peak_yx(src_mom0, kinms_grid.grid_2d)
    print(f"  Source mom0 peak: ({sy:+.4f}, {sx:+.4f})")

    lens_mom0 = lensed_cube.sum(axis=0) * z_step
    ly, lx = peak_yx(lens_mom0, image_grid.grid_2d)
    print(f"  Lensed model mom0 peak: ({ly:+.4f}, {lx:+.4f})")

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis_instance.masked_dataset.data,
        transformers=analysis_instance.transformers,
    )
    dy, dx = peak_yx(dirty_data.sum(axis=0) * z_step, image_grid.grid_2d)
    print(f"  Dirty data mom0 peak: ({dy:+.4f}, {dx:+.4f})")
    print(
        f"  Model − data mom0 peak offset: ({ly - dy:+.4f}, {lx - dx:+.4f}) arcsec"
    )

    inst = analysis_instance.masked_dataset.instance
    src_res = _source_plane_resolution_note(
        kinms_grid_3d=kinms_grid,
        truth_params=truth_params,
        kinms_instance=inst,
    )
    extent_y, extent_x, src_peak = _source_mom0_extent_arcsec(
        source_cube, kinms_grid, z_step
    )
    print("\n  Source-plane size (effective_radius in arcsec, not pixels):")
    print(
        f"    profile effective_radius = {src_res['effective_radius_arcsec']:.4g}″ "
        f"(intensity = {truth_params.get('intensity', 'n/a')} Jy km/s)"
    )
    print(
        f"    source pixel scale = {src_res['source_pixel_scale_arcsec']:.6g}″ "
        f"({src_res['source_n_pixels']}² grid)"
    )
    print(
        f"    pixels across effective_radius ≈ "
        f"{src_res['pixels_across_effective_radius']:.2f}"
    )
    if inst is not None and getattr(inst, "obj", None) is not None:
        print(
            f"    KinMS field xs={src_res['kinms_xs_arcsec']:.4g}″, "
            f"ys={src_res['kinms_ys_arcsec']:.4g}″, "
            f"cellSize={src_res['kinms_cell_size_arcsec']:.6g}″"
        )
        print(
            f"    grid extent y={src_res['grid_y_extent_arcsec']:.4g}″, "
            f"x={src_res['grid_x_extent_arcsec']:.4g}″ "
            f"(cellSize match={src_res['cell_size_matches_grid']})"
        )
    print(
        f"    source mom0 {int(0.2*100)}% peak extent ≈ "
        f"{extent_y:.4g}″ × {extent_x:.4g}″ "
        f"(projected major-axis scale re/cos(incl) ≈ "
        f"{src_res['projected_major_axis_arcsec']:.4g}″)"
    )
    lens_ey, lens_ex, _ = _source_mom0_extent_arcsec(
        lensed_cube, image_grid, z_step
    )
    data_ey, data_ex, _ = _source_mom0_extent_arcsec(
        dirty_data, image_grid, z_step
    )
    print(
        f"    lensed model mom0 20% peak extent ≈ "
        f"{lens_ey:.4g}″ × {lens_ex:.4g}″"
    )
    print(
        f"    dirty data mom0 20% peak extent ≈ "
        f"{data_ey:.4g}″ × {data_ex:.4g}″"
    )
    if not src_res["resolves_effective_radius"]:
        print(
            "    SOURCE UNDERSAMPLED — effective_radius spans fewer than 5 source "
            "pixels; add a source_grid bounding box or raise source_grid.n_pixels."
        )

    pa_src = _mom0_major_axis_pa_from_mom0(src_mom0, kinms_grid.grid_2d)
    pa_lens = _mom0_major_axis_pa_from_mom0(lens_mom0, image_grid.grid_2d)
    pa_dirty = _mom0_major_axis_pa_from_mom0(
        dirty_data.sum(axis=0) * z_step, image_grid.grid_2d
    )
    print(
        f"  Position angles (mom0 major axis): source={pa_src:.1f}°, "
        f"lensed model={pa_lens:.1f}°, dirty data={pa_dirty:.1f}° "
        f"(truth phi={truth_params.get('phi', float('nan')):.1f}°)"
    )


def _print_flux_conservation_report(report, rtol=1e-2):
    print("\n=== Flux (source plane vs lensed image plane) ===\n")
    print(
        f"  source grid = {report['source_n_pixels']} px "
        f"({report['source_pixel_area_arcsec2']:.4g} arcsec²/pix)"
    )
    print(
        f"  image grid  = {report['image_n_pixels']} px "
        f"({report['image_pixel_area_arcsec2']:.4g} arcsec²/pix)"
    )
    print(
        f"  pixel-area scale applied in ray_trace = "
        f"{report['pixel_area_ratio_image_over_source']:.4g}"
    )
    print(
        f"  source line flux  = {report['source_line_flux_jy_kms']:.6g} Jy km/s"
    )
    print(
        f"  lensed line flux  = {report['lensed_line_flux_jy_kms']:.6g} Jy km/s"
    )
    print(
        f"  lensed / source line flux = {report['line_flux_ratio']:.4f} "
        "(differences expected from lensing)"
    )

    traced_frac = report.get("traced_inside_source_bbox_fraction")
    if traced_frac is not None:
        print(
            f"\n  Traced image-plane rays inside source grid = {traced_frac:.1%}"
        )
        if traced_frac < 0.95:
            print(
                "    SOURCE BBOX CLIPPING — widen source_grid xmin/xmax/ymin/ymax "
                "(or use a full-field source_grid) so back-traced rays are not "
                "zeroed at the source-plane boundary."
            )

    rebin_ok = True
    if report.get("rebin_line_flux_relative_error") is not None:
        rebin_err = report["rebin_line_flux_relative_error"]
        print(
            f"\n  Rebin check (fine lensed → coarse image grid):"
        )
        print(
            f"    fine lensed line flux   = "
            f"{report['fine_lensed_line_flux_jy_kms']:.6g} Jy km/s"
        )
        print(
            f"    coarse lensed line flux = "
            f"{report['lensed_line_flux_jy_kms']:.6g} Jy km/s"
        )
        print(f"    relative error = {rebin_err:+.4%}")
        rebin_ok = np.isfinite(rebin_err) and abs(rebin_err) <= rtol
        if rebin_ok:
            print(
                f"    REBIN PASS — flux conserved within {rtol:.0%} when "
                "downsampling the lensed cube."
            )
        else:
            print(
                f"    REBIN FAIL — coarse lensed flux differs from fine "
                f"lensed by more than {rtol:.0%}."
            )

    return rebin_ok


def _mom0_image_stats(data_mom0, model_mom0):
    data_mom0 = np.asarray(data_mom0, dtype=float)
    model_mom0 = np.asarray(model_mom0, dtype=float)
    residual = data_mom0 - model_mom0
    data_rms = float(np.sqrt(np.mean(data_mom0**2)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    return {
        "data_rms": data_rms,
        "residual_rms": residual_rms,
        "fractional_residual_rms": residual_rms / data_rms if data_rms > 0 else float("nan"),
    }


def _image_plane_resolution_note(settings, truth_params):
    img_n, img_pix, img_width = autolens_utils.image_plane_grid_from_settings(settings)
    re = float(truth_params.get("effective_radius", 0.05))
    pixels_across_re = re / img_pix
    return {
        "image_n_pixels": img_n,
        "image_pixel_scale_arcsec": img_pix,
        "effective_radius_arcsec": re,
        "pixels_across_effective_radius": pixels_across_re,
        "undersampled_for_dirty_images": pixels_across_re < 3.0,
    }


def _source_plane_resolution_note(kinms_grid_3d, truth_params, kinms_instance=None):
    """Report whether the KinMS grid resolves effective_radius in arcsec."""
    re = float(truth_params.get("effective_radius", 0.05))
    incl = float(truth_params.get("inclination", 65.0))
    pix = float(kinms_grid_3d.pixel_scale)
    pixels_across_re = re / pix
    projected_major = re / np.cos(np.radians(incl))

    note = {
        "source_n_pixels": kinms_grid_3d.n_pixels,
        "source_pixel_scale_arcsec": pix,
        "effective_radius_arcsec": re,
        "pixels_across_effective_radius": pixels_across_re,
        "projected_major_axis_arcsec": projected_major,
        "resolves_effective_radius": pixels_across_re >= 5.0,
    }

    if kinms_instance is not None and getattr(kinms_instance, "obj", None) is not None:
        obj = kinms_instance.obj
        note["kinms_xs_arcsec"] = float(obj.xs)
        note["kinms_ys_arcsec"] = float(obj.ys)
        note["kinms_cell_size_arcsec"] = float(obj.cellSize)
        coords = np.asarray(kinms_grid_3d.grid_2d)
        y_extent = float(coords[:, 0].max() - coords[:, 0].min())
        x_extent = float(coords[:, 1].max() - coords[:, 1].min())
        note["grid_y_extent_arcsec"] = y_extent
        note["grid_x_extent_arcsec"] = x_extent
        note["cell_size_matches_grid"] = np.isclose(
            obj.cellSize, x_extent / kinms_grid_3d.n_pixels, rtol=0.02
        )

    return note


def _mom0_major_axis_pa_from_mom0(mom0, grid_2d, threshold=0.2):
    mom0 = np.asarray(mom0, dtype=float)
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    y = coords[:, :, 0]
    x = coords[:, :, 1]
    peak = float(np.nanmax(mom0))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    w = np.where(mom0 >= threshold * peak, mom0, 0.0)
    if w.sum() <= 0:
        return float("nan")
    yc = (w * y).sum() / w.sum()
    xc = (w * x).sum() / w.sum()
    dy = y - yc
    dx = x - xc
    mu20 = (w * dx * dx).sum() / w.sum()
    mu02 = (w * dy * dy).sum() / w.sum()
    mu11 = (w * dx * dy).sum() / w.sum()
    return float(np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)))


def _source_mom0_extent_arcsec(source_cube, kinms_grid_3d, z_step_kms, level=0.2):
    mom0 = np.asarray(source_cube, dtype=float).sum(axis=0) * z_step_kms
    coords = np.asarray(kinms_grid_3d.grid_2d).reshape(kinms_grid_3d.shape_2d + (2,))
    peak = float(np.nanmax(mom0))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan"), float("nan"), peak
    mask = mom0 >= level * peak
    if not mask.any():
        return float("nan"), float("nan"), peak
    ys = coords[:, :, 0][mask]
    xs = coords[:, :, 1][mask]
    return float(ys.max() - ys.min()), float(xs.max() - xs.min()), peak


def _dirty_cube_on_image_grid(visibilities, uv_wavelengths, n_pixels, real_space_width, n_channels):
    pixel_scale = real_space_width / n_pixels
    mask_3d = Mask3D.unmasked(
        n_channels=n_channels,
        shape_2d=(n_pixels, n_pixels),
        pixel_scales=pixel_scale,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
    )
    return autolens_utils.dirty_cube_from(
        visibilities=visibilities,
        transformers=transformers,
    )


def _print_source_params(instance):
    src = instance.galaxies.source
    print("\n=== Truth source parameters (named) ===\n")
    for key in sorted(k for k in src.__dict__ if not k.startswith("_")):
        if key in ("id", "cls"):
            continue
        print(f"  {key} = {getattr(src, key)!r}")


def _plots_output_dir(settings, plots_dir=None):
    if plots_dir is not None:
        return Path(plots_dir)
    return Path(settings.get("output_path", ".")) / "truth_model_test"


def _source_plane_plot_extent(settings, kinms_grid_3d, phase1_result):
    if phase1_result is not None:
        bbox = reconstruction.source_plane_bounding_box_from_result(phase1_result)
    else:
        bbox = autolens_utils.source_grid_bounding_box_from_cfg(
            settings.get("source_grid", {})
        )
    if bbox is not None:
        return autolens_utils.image_extent_from_bounding_box(bbox)

    coords = np.asarray(kinms_grid_3d.grid_2d)
    return (
        float(coords[:, 1].min()),
        float(coords[:, 1].max()),
        float(coords[:, 0].min()),
        float(coords[:, 0].max()),
    )


def _save_truth_fit_plots(
    analysis_instance,
    model_data,
    source_cube,
    lensed_cube,
    settings,
    pipeline,
    output_dir,
    plot_n_pixels=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = output_dir / "fit_dataset"
    fit_dir.mkdir(parents=True, exist_ok=True)

    vis = visualizer_mod.VisualizerAbstract(
        masked_dataset=analysis_instance.masked_dataset,
        transformers=analysis_instance.transformers,
        directory=str(fit_dir),
    )
    vis.visualize_data()
    vis.visualize(model_data=model_data, during_analysis=False)

    dirty_data_cube = vis.dirty_cube
    dirty_model_cube = autolens_utils.dirty_cube_from(
        visibilities=model_data,
        transformers=analysis_instance.transformers,
    )
    z_step = analysis_instance.masked_dataset.z_step_kms

    img_extent = autolens_utils.image_extent_arcsec(settings["real_space_width"])
    src_extent = _source_plane_plot_extent(
        settings,
        pipeline["kinms_grid_3d"],
        pipeline["phase1_result"],
    )

    save_cube(
        dirty_data_cube,
        output_dir / "dirty_data_cube.png",
        extent=img_extent,
    )
    save_cube(
        dirty_model_cube,
        output_dir / "dirty_model_cube.png",
        extent=img_extent,
    )

    dirty_mom0_data = dirty_data_cube.sum(axis=0) * z_step
    dirty_mom0_model = dirty_model_cube.sum(axis=0) * z_step
    mom0_residual = dirty_mom0_data - dirty_mom0_model
    save_fit_triplet(
        data=dirty_mom0_data,
        model=dirty_mom0_model,
        residuals=mom0_residual,
        path=output_dir / "fit_triplet_mom0.png",
        titles=("Dirty data (mom0)", "Dirty model (mom0)", "Residuals (mom0)"),
        extent=img_extent,
        scale_mode="data",
    )
    save_fit_triplet(
        data=dirty_mom0_data,
        model=dirty_mom0_model,
        residuals=mom0_residual,
        path=output_dir / "fit_triplet_mom0_residual_focus.png",
        titles=("Dirty data (mom0)", "Dirty model (mom0)", "Residuals (mom0)"),
        extent=img_extent,
        scale_mode="residual",
    )

    source_mom0 = source_cube.sum(axis=0) * z_step
    lensed_mom0 = lensed_cube.sum(axis=0) * z_step
    save_image(
        source_mom0,
        output_dir / "source_plane_mom0.png",
        title="Source-plane KinMS mom0 (Jy km/s/pixel)",
        extent=src_extent,
    )
    save_image(
        lensed_mom0,
        output_dir / "image_plane_mom0.png",
        title="Image-plane lensed mom0 (Jy km/s/pixel)",
        extent=img_extent,
    )

    if plot_n_pixels is None:
        plot_n_pixels = int(settings.get("source_grid", {}).get("plot_n_pixels", 160))
    if plot_n_pixels > settings["n_pixels"]:
        uv_wavelengths = analysis_instance.masked_dataset.uv_wavelengths
        n_channels = analysis_instance.masked_dataset.n_channels
        data_vis = analysis_instance.masked_dataset.data
        hi_data = _dirty_cube_on_image_grid(
            data_vis,
            uv_wavelengths,
            n_pixels=plot_n_pixels,
            real_space_width=settings["real_space_width"],
            n_channels=n_channels,
        )
        hi_model = _dirty_cube_on_image_grid(
            model_data,
            uv_wavelengths,
            n_pixels=plot_n_pixels,
            real_space_width=settings["real_space_width"],
            n_channels=n_channels,
        )
        hi_mom0_data = hi_data.sum(axis=0) * z_step
        hi_mom0_model = hi_model.sum(axis=0) * z_step
        hi_residual = hi_mom0_data - hi_mom0_model
        hi_extent = autolens_utils.image_extent_arcsec(settings["real_space_width"])
        save_fit_triplet(
            data=hi_mom0_data,
            model=hi_mom0_model,
            residuals=hi_residual,
            path=output_dir / f"fit_triplet_mom0_{plot_n_pixels}.png",
            titles=(
                f"Dirty data mom0 ({plot_n_pixels}²)",
                f"Dirty model mom0 ({plot_n_pixels}²)",
                "Residuals",
            ),
            extent=hi_extent,
            scale_mode="data",
        )

    print(f"\n=== Plots saved to {output_dir.resolve()} ===\n")
    print(f"  {fit_dir / 'data.png'}")
    print(f"  {fit_dir / 'model.png'}")
    print(f"  {fit_dir / 'residuals.png'}")
    print(f"  {output_dir / 'fit_triplet_mom0.png'}")
    print(f"  {output_dir / 'fit_triplet_mom0_residual_focus.png'}")
    print(f"  {output_dir / 'source_plane_mom0.png'}")
    print(f"  {output_dir / 'image_plane_mom0.png'}")
    if plot_n_pixels > settings["n_pixels"]:
        print(f"  {output_dir / f'fit_triplet_mom0_{plot_n_pixels}.png'}")
    print()


def _build_pipeline(settings, run_phase1=False, lens_centre_mode="fixed"):
    mode = validate_normalization_settings(settings)
    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)

    img_n, img_pix, img_width = autolens_utils.image_plane_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=img_n,
        pixel_scale=img_pix,
        n_channels=len(frequencies),
    )

    phase1_result = None
    sb_map = None
    int_flux = None
    if requires_phase1(mode):
        if not run_phase1:
            raise ValueError(
                f"normalization_mode='{mode}' requires --run-phase1 for a truth test."
            )
        phase1_result = reconstruction.run_reconstruction(settings)
        kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
            settings,
            n_channels=len(frequencies),
            phase1_result=phase1_result,
        )
        flux_snr_threshold = reconstruction.flux_snr_threshold_from_settings(settings)
        sb_map, lens_centre_phase1, _sb_full, _noise = (
            reconstruction.source_sb_for_phase2_flux(
                result=phase1_result,
                grid_2d=kinms_grid_3d.grid_2d,
                snr_threshold=flux_snr_threshold,
            )
        )
        sb_input_units = settings.get("reconstruction", {}).get(
            "sb_input_units", "jy_per_pixel_per_channel"
        )
        if mode == PIXELIZED:
            dataset_instance = kinms_utils.make_pixelized_instance_from_grid(
                grid_3d=kinms_grid_3d,
                z_step_kms=z_step_kms,
                sb_map=sb_map,
                sb_input_units=sb_input_units,
                **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
            )
        else:
            int_flux = kinms_utils.kinms_intflux_from_sb_map(
                sb_map=sb_map,
                z_step_kms=z_step_kms,
                n_channels=kinms_grid_3d.n_channels,
                sb_input_units=sb_input_units,
            )
            dataset_instance = kinms_utils.make_instance_from_grid(
                grid_3d=kinms_grid_3d,
                z_step_kms=z_step_kms,
                attach_grid=True,
                disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
            )
            dataset_instance.int_flux = int_flux
    else:
        kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
            settings, n_channels=len(frequencies)
        )
        dataset_instance = kinms_utils.make_instance_from_grid(
            grid_3d=kinms_grid_3d,
            z_step_kms=z_step_kms,
            attach_grid=True,
            disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
        )
        lens_centre_phase1 = None

    lens_fixed = (
        settings["lens_mass_model"]["centre_0"],
        settings["lens_mass_model"]["centre_1"],
    )
    if lens_centre_mode == "phase1":
        if lens_centre_phase1 is None:
            raise ValueError("--lens-centre phase1 requires --run-phase1.")
        lens_centre = lens_centre_phase1
    else:
        lens_centre = lens_fixed

    mask_3d = Mask3D.unmasked(
        n_channels=image_grid_3d.n_channels,
        shape_2d=image_grid_3d.shape_2d,
        pixel_scales=image_grid_3d.pixel_scales,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
        settings=settings,
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    tracer = build_tracer(settings, centre=lens_centre)
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
        instance=dataset_instance,
    )
    analysis_instance = analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )

    return {
        "mode": mode,
        "analysis": analysis_instance,
        "frequencies": frequencies,
        "z_step_kms": z_step_kms,
        "kinms_grid_3d": kinms_grid_3d,
        "image_grid_3d": image_grid_3d,
        "lens_centre": lens_centre,
        "lens_centre_fixed": lens_fixed,
        "lens_centre_phase1": lens_centre_phase1,
        "int_flux": int_flux,
        "phase1_result": phase1_result,
    }


def run_truth_test(
    settings,
    run_phase1=False,
    lens_centre_mode="fixed",
    save_plots=True,
    plots_dir=None,
    optimize_intensity=False,
    optimize_intensity_and_effective_radius=False,
    optimize_lens_centre=False,
    intensity_lower=None,
    intensity_upper=None,
    effective_radius_lower=None,
    effective_radius_upper=None,
    optimization_start_from="prior_center",
    lens_centre_half_width=0.05,
    lens_centre_0_lower=None,
    lens_centre_0_upper=None,
    lens_centre_1_lower=None,
    lens_centre_1_upper=None,
):
    pipeline = _build_pipeline(
        settings,
        run_phase1=run_phase1,
        lens_centre_mode=lens_centre_mode,
    )
    analysis_instance = pipeline["analysis"]
    truth_params = truth_parameters_from_settings(settings)
    instance = truth_instance_from_settings(settings)

    if pipeline["int_flux"] is not None:
        truth_params["intensity"] = pipeline["int_flux"]

    print("\n=== Truth model test (no optimization) ===\n")
    print(f"  normalization_mode = {pipeline['mode']}")
    print(f"  model_name = {settings.get('model_name', 'KinMS')}")
    print(f"  z_step_kms = {pipeline['z_step_kms']:.4f}")
    print(
        f"  KinMS source grid = "
        f"{autolens_utils.source_grid_label_from_settings(settings, pipeline['phase1_result'])}"
    )
    print(
        f"  Image plane grid = {pipeline['image_grid_3d'].n_pixels}² "
        f"({settings['real_space_width']}″ field)"
    )
    print(f"  Lens centre = {pipeline['lens_centre']} ({lens_centre_mode})")
    if pipeline["lens_centre_phase1"] is not None:
        dc = pipeline["lens_centre_phase1"]
        df = pipeline["lens_centre_fixed"]
        print(
            f"  Phase-1 − fixed lens Δcentre = "
            f"({dc[0]-df[0]:+.4f}, {dc[1]-df[1]:+.4f}) arcsec"
        )
    if pipeline["int_flux"] is not None:
        print(f"  intFlux from phase 1 = {pipeline['int_flux']:.4f} Jy km/s")

    print("\n=== Truth parameters (from settings) ===\n")
    for key in PARAMETRIC_KINMS_DEBUG_ORDER:
        if key in truth_params:
            print(f"  {key} = {truth_params[key]!r}")

    _print_source_params(instance)

    flip_vel = bool(settings.get("flip_velocity_axis", False))
    flip_y = bool(settings.get("flip_kinms_y_before_lensing", True))
    print("\n=== Convention flags ===\n")
    print(f"  flip_velocity_axis          = {flip_vel}")
    print(f"  flip_kinms_y_before_lensing = {flip_y}")
    print(
        f"  Analysis._flip_velocity_axis = "
        f"{getattr(analysis_instance, '_flip_velocity_axis', None)}"
    )

    print("\n=== Model construction steps ===\n")

    source_cube = analysis_instance.model_cube_from_instance(instance=instance)
    _finite_stats(source_cube, "1. Source-plane KinMS cube")
    # Prove spectral flip: first vs last channel must differ for an inclined disk,
    # and toggling the flag must swap them.
    ch0 = float(np.nansum(source_cube[0]))
    ch_last = float(np.nansum(source_cube[-1]))
    print(
        f"  Channel flux sums: ch[0]={ch0:.6g}, ch[-1]={ch_last:.6g}, "
        f"Δ={ch0 - ch_last:+.6g}"
    )
    if flip_vel:
        print(
            "  (flip_velocity_axis=True: model cube channels were reversed "
            "after KinMS)"
        )
    print(
        "  Note: moment-0 maps are unchanged by a pure spectral reverse; "
        "compare mom1 / channel maps or UV chi²."
    )

    # Side-by-side UV likelihood: flipped vs unflipped (same instance).
    model_data_on = analysis_instance.model_data_from_instance(instance=instance)
    stats_on = _visibility_fit_stats(analysis_instance, model_data_on)
    analysis_instance._flip_velocity_axis = not flip_vel
    model_data_off = analysis_instance.model_data_from_instance(instance=instance)
    stats_off = _visibility_fit_stats(analysis_instance, model_data_off)
    analysis_instance._flip_velocity_axis = flip_vel  # restore settings value
    print("\n=== Velocity-flip UV check (same truth instance) ===\n")
    print(
        f"  flip_velocity_axis={flip_vel} (settings): "
        f"chi²/N={stats_on['chi_squared_per_datum']:.6f}, "
        f"logL={stats_on['log_likelihood']:.2f}"
    )
    print(
        f"  flip_velocity_axis={not flip_vel} (toggled): "
        f"chi²/N={stats_off['chi_squared_per_datum']:.6f}, "
        f"logL={stats_off['log_likelihood']:.2f}"
    )
    dchi = stats_on["chi_squared_per_datum"] - stats_off["chi_squared_per_datum"]
    print(f"  Δ(chi²/N) settings − toggled = {dchi:+.6f}")
    if abs(dchi) < 1.0e-9:
        print(
            "  WARNING: chi² unchanged — flip had no effect on UV likelihood "
            "(check Analysis settings wiring, or spectrum is symmetric)."
        )
    else:
        print("  Velocity flip is affecting the UV fit (as expected for kinematics).")

    from src.utils import analysis_utils

    lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=source_cube,
        tracer=analysis_instance.tracer,
        grid=analysis_instance.masked_dataset.grid_3d.grid_2d,
        z_mask=analysis_instance.masked_dataset.mask_3d.z_mask,
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
        output_shape=analysis_instance.masked_dataset.grid_3d.shape_2d,
    )
    _finite_stats(lensed_cube, "2. Image-plane lensed cube")

    source_cfg = settings.get("source_grid", {})
    fine_n = int(source_cfg.get("n_pixels", settings["n_pixels"]))
    fine_image_grid = Grid3D.uniform(
        n_pixels=fine_n,
        pixel_scale=settings["real_space_width"] / fine_n,
        n_channels=pipeline["image_grid_3d"].n_channels,
    )
    lensed_cube_fine = analysis_utils.lensed_cube_on_grid(
        cube=source_cube,
        tracer=analysis_instance.tracer,
        image_grid_2d=fine_image_grid.grid_2d,
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
    )
    lensed_cube_rebinned = np.stack(
        [
            analysis_utils.resample_image_to_shape_flux_conserving(
                lensed_cube_fine[i],
                analysis_instance.masked_dataset.grid_3d.shape_2d,
            )
            for i in range(lensed_cube_fine.shape[0])
        ],
        axis=0,
    )

    traced_source_coords = np.asarray(
        analysis_instance.tracer.traced_grid_2d_list_from(
            grid=analysis_instance.masked_dataset.grid_3d.grid_2d
        )[1]
    )
    flux_report = analysis_utils.flux_conservation_lensing_report(
        source_cube=source_cube,
        lensed_cube=lensed_cube,
        z_step_kms=pipeline["z_step_kms"],
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
        image_grid_2d=pipeline["image_grid_3d"].grid_2d,
        traced_source_coords=traced_source_coords,
        lensed_cube_fine=lensed_cube_fine,
        fine_image_grid_2d=fine_image_grid.grid_2d,
    )
    flux_pass = _print_flux_conservation_report(flux_report)

    _print_coordinate_audit(
        settings=settings,
        pipeline=pipeline,
        source_cube=source_cube,
        lensed_cube=lensed_cube,
        analysis_instance=analysis_instance,
        truth_params=truth_params,
    )

    rebin_direct_err = (
        (lensed_cube.sum() - lensed_cube_rebinned.sum()) / lensed_cube_rebinned.sum()
        if lensed_cube_rebinned.sum() > 0
        else float("nan")
    )
    print(
        f"  one-step coarse vs two-step rebin relative error = "
        f"{rebin_direct_err:+.4%}\n"
    )

    model_data = analysis_instance.model_data_from_instance(instance=instance)
    _finite_stats(model_data, "3. Model visibilities (real/imag)")

    stats = _visibility_fit_stats(analysis_instance, model_data)

    dirty_data_cube = autolens_utils.dirty_cube_from(
        visibilities=analysis_instance.masked_dataset.data,
        transformers=analysis_instance.transformers,
    )
    dirty_model_cube = autolens_utils.dirty_cube_from(
        visibilities=model_data,
        transformers=analysis_instance.transformers,
    )
    z_step = pipeline["z_step_kms"]
    mom0_stats = _mom0_image_stats(
        dirty_data_cube.sum(axis=0) * z_step,
        dirty_model_cube.sum(axis=0) * z_step,
    )
    resolution = _image_plane_resolution_note(settings, truth_params)

    print("\n=== Fit at truth (no optimization) ===\n")
    print(f"  log likelihood = {stats['log_likelihood']:.2f}")
    print(f"  chi-squared = {stats['chi_squared']:.2f}")
    print(f"  chi-squared / N = {stats['chi_squared_per_datum']:.4f}")
    print(f"  normalized residual RMS = {stats['normalized_residual_rms']:.4f}")
    print(f"  residual RMS = {stats['residual_rms']:.4g} Jy")
    print(f"  median sigma = {stats['median_sigma']:.4g} Jy")
    print(f"  N visibility samples = {stats['n_data']}")

    print("\n=== Dirty moment-0 map (image plane, fit grid) ===\n")
    print(f"  mom0 data RMS = {mom0_stats['data_rms']:.4g}")
    print(f"  mom0 residual RMS = {mom0_stats['residual_rms']:.4g}")
    print(
        f"  mom0 fractional residual RMS = "
        f"{mom0_stats['fractional_residual_rms']:.4g}"
    )

    print("\n=== Image-plane resolution (dirty-plot context) ===\n")
    print(
        f"  image pixel scale = {resolution['image_pixel_scale_arcsec']:.4g}″ "
        f"({resolution['image_n_pixels']}² over {settings['real_space_width']}″)"
    )
    print(f"  effective_radius = {resolution['effective_radius_arcsec']:.4g}″")
    print(
        f"  pixels across effective_radius ≈ "
        f"{resolution['pixels_across_effective_radius']:.2f}"
    )

    visibility_pass = (
        np.isfinite(stats["log_likelihood"])
        and 0.5 <= stats["chi_squared_per_datum"] <= 2.0
    )
    mom0_pass = mom0_stats["fractional_residual_rms"] < 0.5

    print("\n=== Verdict ===\n")
    if visibility_pass:
        print(
            "  VISIBILITY PASS — chi-squared / N is near unity in UV space "
            "(this is what the likelihood uses)."
        )
    else:
        print(
            "  VISIBILITY FAIL — chi-squared / N is not near unity; "
            "check parameters, grids, lens centre, or z_step_kms."
        )

    if resolution["undersampled_for_dirty_images"]:
        print(
            "  DIRTY-IMAGE WARNING — the fit image plane is much coarser than "
            "the source effective radius, so 40² dirty maps are poor visual "
            "diagnostics even when the UV fit is good. Inspect "
            "fit_triplet_mom0_<hi-res>.png and source_plane_mom0.png instead."
        )

    if mom0_pass:
        print("  MOM0 PASS — dirty moment-0 residual is small on the fit grid.")
    else:
        print(
            "  MOM0 FAIL — dirty moment-0 residual is large; morphology or "
            "flux may be wrong on the image plane."
        )

    ok = visibility_pass
    print()

    if save_plots:
        _save_truth_fit_plots(
            analysis_instance=analysis_instance,
            model_data=model_data,
            source_cube=source_cube,
            lensed_cube=lensed_cube,
            settings=settings,
            pipeline=pipeline,
            output_dir=_plots_output_dir(settings, plots_dir=plots_dir),
        )

    stats["pass"] = ok
    stats["visibility_pass"] = visibility_pass
    stats["mom0_pass"] = mom0_pass
    stats["flux_pass"] = flux_pass
    stats["flux_report"] = flux_report
    stats["mom0_stats"] = mom0_stats
    stats["resolution"] = resolution

    if optimize_intensity:
        truth_intensity = float(
            truth_params.get("intensity", instance.galaxies.source.intensity)
        )
        intensity_report = _optimize_intensity_at_truth(
            settings=settings,
            analysis_instance=analysis_instance,
            truth_intensity=truth_intensity,
            intensity_lower=intensity_lower,
            intensity_upper=intensity_upper,
        )
        _print_intensity_optimization_report(intensity_report)
        stats["intensity_optimization"] = intensity_report
        if save_plots:
            opt_dir = _plots_output_dir(settings, plots_dir=plots_dir) / "optimized_intensity"
            opt_instance = intensity_report["best_instance"]
            opt_source_cube = analysis_instance.model_cube_from_instance(
                instance=opt_instance
            )
            opt_lensed_cube = analysis_utils.lensed_cube_from_tracer(
                cube=opt_source_cube,
                tracer=analysis_instance.tracer,
                grid=analysis_instance.masked_dataset.grid_3d.grid_2d,
                z_mask=analysis_instance.masked_dataset.mask_3d.z_mask,
                source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
                output_shape=analysis_instance.masked_dataset.grid_3d.shape_2d,
            )
            _save_truth_fit_plots(
                analysis_instance=analysis_instance,
                model_data=intensity_report["best_model_data"],
                source_cube=opt_source_cube,
                lensed_cube=opt_lensed_cube,
                settings=settings,
                pipeline=pipeline,
                output_dir=opt_dir,
            )
            print(f"\n=== Optimized-intensity plots saved to {opt_dir} ===\n")

    if optimize_lens_centre:
        lens_report = _optimize_lens_centre_at_truth(
            settings=settings,
            pipeline=pipeline,
            truth_lens_centre=pipeline["lens_centre"],
            half_width=lens_centre_half_width,
            centre_0_lower=lens_centre_0_lower,
            centre_0_upper=lens_centre_0_upper,
            centre_1_lower=lens_centre_1_lower,
            centre_1_upper=lens_centre_1_upper,
        )
        _print_lens_centre_optimization_report(lens_report)
        stats["lens_centre_optimization"] = lens_report
        if save_plots:
            opt_dir = (
                _plots_output_dir(settings, plots_dir=plots_dir)
                / "optimized_lens_centre"
            )
            opt_pipeline = dict(pipeline)
            opt_pipeline["analysis"] = lens_report["best_analysis"]
            opt_pipeline["lens_centre"] = lens_report["best_lens_centre"]
            _save_truth_fit_plots(
                analysis_instance=lens_report["best_analysis"],
                model_data=lens_report["best_model_data"],
                source_cube=lens_report["best_source_cube"],
                lensed_cube=lens_report["best_lensed_cube"],
                settings=settings,
                pipeline=opt_pipeline,
                output_dir=opt_dir,
            )
            print(f"\n=== Optimized-lens-centre plots saved to {opt_dir} ===\n")

    if optimize_intensity_and_effective_radius:
        truth_intensity = float(
            truth_params.get("intensity", instance.galaxies.source.intensity)
        )
        truth_effective_radius = float(
            truth_params.get(
                "effective_radius", instance.galaxies.source.effective_radius
            )
        )
        joint_report = _optimize_intensity_and_effective_radius_at_truth(
            settings=settings,
            analysis_instance=analysis_instance,
            truth_intensity=truth_intensity,
            truth_effective_radius=truth_effective_radius,
            intensity_lower=intensity_lower,
            intensity_upper=intensity_upper,
            effective_radius_lower=effective_radius_lower,
            effective_radius_upper=effective_radius_upper,
            start_from=optimization_start_from,
        )
        _print_intensity_and_effective_radius_optimization_report(joint_report)
        stats["intensity_and_effective_radius_optimization"] = joint_report
        if save_plots:
            opt_dir = (
                _plots_output_dir(settings, plots_dir=plots_dir)
                / "optimized_intensity_and_re"
            )
            opt_instance = joint_report["best_instance"]
            opt_source_cube = analysis_instance.model_cube_from_instance(
                instance=opt_instance
            )
            opt_lensed_cube = analysis_utils.lensed_cube_from_tracer(
                cube=opt_source_cube,
                tracer=analysis_instance.tracer,
                grid=analysis_instance.masked_dataset.grid_3d.grid_2d,
                z_mask=analysis_instance.masked_dataset.mask_3d.z_mask,
                source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
                output_shape=analysis_instance.masked_dataset.grid_3d.shape_2d,
            )
            _save_truth_fit_plots(
                analysis_instance=analysis_instance,
                model_data=joint_report["best_model_data"],
                source_cube=opt_source_cube,
                lensed_cube=opt_lensed_cube,
                settings=settings,
                pipeline=pipeline,
                output_dir=opt_dir,
            )
            print(
                "\n=== Optimized intensity+re plots saved to "
                f"{opt_dir} ===\n"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate truth KinMS model against data without optimization."
    )
    parser.add_argument(
        "--settings",
        required=True,
        help="Path to runner settings JSON.",
    )
    parser.add_argument(
        "--run-phase1",
        action="store_true",
        help="Run phase-1 reconstruction (required for pixelized / flux-from-phase1).",
    )
    parser.add_argument(
        "--lens-centre",
        choices=("fixed", "phase1"),
        default="fixed",
        help="Lens mass centre for ray tracing (default: fixed JSON values).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for output plots (default: <output_path>/truth_model_test/).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating fit plots.",
    )
    parser.add_argument(
        "--optimize-intensity",
        action="store_true",
        help=(
            "After the fixed-truth test, optimize KinMS intFlux (intensity) "
            "with all other parameters held at truth."
        ),
    )
    parser.add_argument(
        "--optimize-intensity-and-re",
        action="store_true",
        help=(
            "After the fixed-truth test, optimize intensity and effective_radius "
            "with all other parameters held at truth."
        ),
    )
    parser.add_argument(
        "--optimize-lens-centre",
        action="store_true",
        help=(
            "After the fixed-truth test, optimize lens mass centre_0/centre_1 "
            "with source parameters held at truth."
        ),
    )
    parser.add_argument(
        "--lens-centre-half-width",
        type=float,
        default=0.05,
        help=(
            "Half-width of lens-centre search box around JSON values (arcsec); "
            "default 0.05."
        ),
    )
    parser.add_argument(
        "--lens-centre-0-lower",
        type=float,
        default=None,
        help="Explicit lower bound for lens centre_0 search (arcsec).",
    )
    parser.add_argument(
        "--lens-centre-0-upper",
        type=float,
        default=None,
        help="Explicit upper bound for lens centre_0 search (arcsec).",
    )
    parser.add_argument(
        "--lens-centre-1-lower",
        type=float,
        default=None,
        help="Explicit lower bound for lens centre_1 search (arcsec).",
    )
    parser.add_argument(
        "--lens-centre-1-upper",
        type=float,
        default=None,
        help="Explicit upper bound for lens centre_1 search (arcsec).",
    )
    parser.add_argument(
        "--optimization-start-from",
        choices=("prior_center", "truth"),
        default="prior_center",
        help=(
            "Starting point for intensity / re optimization "
            "(default: geometric centre of prior bounds)."
        ),
    )
    parser.add_argument(
        "--intensity-lower",
        type=float,
        default=None,
        help="Lower bound for intensity search (Jy km/s); default from priors.",
    )
    parser.add_argument(
        "--intensity-upper",
        type=float,
        default=None,
        help="Upper bound for intensity search (Jy km/s); default from priors.",
    )
    parser.add_argument(
        "--effective-radius-lower",
        type=float,
        default=None,
        help="Lower bound for effective_radius search (arcsec); default from priors.",
    )
    parser.add_argument(
        "--effective-radius-upper",
        type=float,
        default=None,
        help="Upper bound for effective_radius search (arcsec); default from priors.",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    stats = run_truth_test(
        settings,
        run_phase1=args.run_phase1,
        lens_centre_mode=args.lens_centre,
        save_plots=not args.no_plots,
        plots_dir=args.plots_dir,
        optimize_intensity=args.optimize_intensity,
        optimize_intensity_and_effective_radius=args.optimize_intensity_and_re,
        optimize_lens_centre=args.optimize_lens_centre,
        intensity_lower=args.intensity_lower,
        intensity_upper=args.intensity_upper,
        effective_radius_lower=args.effective_radius_lower,
        effective_radius_upper=args.effective_radius_upper,
        optimization_start_from=args.optimization_start_from,
        lens_centre_half_width=args.lens_centre_half_width,
        lens_centre_0_lower=args.lens_centre_0_lower,
        lens_centre_0_upper=args.lens_centre_0_upper,
        lens_centre_1_lower=args.lens_centre_1_lower,
        lens_centre_1_upper=args.lens_centre_1_upper,
    )
    if not np.isfinite(stats["log_likelihood"]) or not stats.get("pass", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
