#!/usr/bin/env python3
"""
Phase-2 pixelized test: truth kinematics, free source centre only.

Workflow:
  1. Phase 1 — pixelized SB reconstruction with **free lens centre**
     (``reconstruction.fix_lens: false``).
  2. Phase 2 — KinMSPixelized cube using phase-1 SB map and phase-1 lens
     centre; **kinematics fixed at truth**, only ``centre_0`` / ``centre_1``
     optimized (direct scipy search by default, optional Nautilus).

Examples:
  python scripts/test_phase2_pixelization.py \\
    --settings settings/runners/kinms_mock_pixelized_truth_kinematics.json

  python scripts/test_phase2_pixelization.py \\
    --settings settings/runners/kinms_mock_pixelized_truth_kinematics.json \\
    --run-nautilus

  python scripts/test_phase2_pixelization.py \\
    --settings settings/runners/kinms_mock_pixelized_truth_kinematics.json \\
    --phase1-only
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup

setup(__file__)

import autofit as af
import numpy as np
from scipy import optimize as scipy_optimize

from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.model import profiles
from src.pipelines import reconstruction
from src.pipelines.normalization import PIXELIZED, priors_for_normalization_mode
from src.pipelines.pixelized_plots import plot_phase1_fit, plot_phase2_fit
from src.pipelines.priors import source_model_from_profile
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.search import build_search_from_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.pipelines.truth_parameters import (
    PIXELIZED_KINEMATIC_KEYS,
    truth_parameters_from_settings,
)
from src.utils import autolens_utils, kinms_utils, spectral_utils


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
    return {
        "log_likelihood": float(fit.likelihood),
        "chi_squared": chi2,
        "chi_squared_per_datum": chi2 / n_data if n_data else float("nan"),
        "n_data": n_data,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "median_sigma": float(np.median(sigma)),
        "normalized_residual_rms": float(
            np.sqrt(np.mean((residual / sigma) ** 2))
        ),
    }


def _output_dir(settings, subdir=None):
    root = Path(settings.get("output_path", ".")) / "phase2_test"
    return root if subdir is None else root / subdir


def run_phase1(settings, output_dir=None):
    """Phase 1: pixelized SB + free lens centre."""
    settings = copy.deepcopy(settings)
    if output_dir is not None:
        settings["output_path"] = str(output_dir)
    settings.setdefault("reconstruction", {})
    if settings["reconstruction"].get("fix_lens", True):
        print(
            "  WARN — reconstruction.fix_lens is true; "
            "set false to free the lens centre in phase 1."
        )

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    print("\n=== Phase 1: pixelized reconstruction (free lens centre) ===\n")
    result = reconstruction.run_reconstruction(settings)

    frequencies, _, _, _ = load_cube_data(settings)
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
        phase1_result=result,
    )
    sb_map, lens_centre = reconstruction.source_sb_on_grid(
        result=result,
        grid_2d=source_grid_3d.grid_2d,
    )

    instance = result.max_log_likelihood_instance
    fitted_lens = instance.galaxies.lens.mass.centre
    mass_cfg = settings["lens_mass_model"]
    print(f"  Fixed JSON lens centre     = ({mass_cfg['centre_0']}, {mass_cfg['centre_1']})")
    print(f"  Phase-1 fitted lens centre = ({fitted_lens[0]:.4f}, {fitted_lens[1]:.4f})")
    print(
        f"  Δ lens centre (fit − JSON) = "
        f"({fitted_lens[0] - mass_cfg['centre_0']:+.4f}, "
        f"{fitted_lens[1] - mass_cfg['centre_1']:+.4f}) arcsec"
    )
    print(f"  Lens centre passed to phase 2 = ({lens_centre[0]:.4f}, {lens_centre[1]:.4f})")

    if output_dir is not None:
        plot_phase1_fit(
            result=result,
            sb_map=sb_map,
            output_dir=Path(output_dir) / "phase1",
            settings=settings,
            sb_map_extent=autolens_utils.image_extent_from_bounding_box(
                reconstruction.source_plane_bounding_box_from_result(result)
            ),
        )

    return {
        "result": result,
        "sb_map": sb_map,
        "lens_centre": lens_centre,
        "source_grid_3d": source_grid_3d,
        "frequencies": frequencies,
    }


def build_phase2_analysis(settings, phase1_bundle):
    """Build phase-2 Analysis with pixelized SB and phase-1 lens centre."""
    frequencies = phase1_bundle["frequencies"]
    sb_map = phase1_bundle["sb_map"]
    lens_centre = phase1_bundle["lens_centre"]
    source_grid_3d = phase1_bundle["source_grid_3d"]

    uv_wavelengths, visibilities, sigma = load_cube_data(settings)[1:]
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)

    img_n, img_pix, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_plane_grid_3d = Grid3D.uniform(
        n_pixels=img_n,
        pixel_scale=img_pix,
        n_channels=len(frequencies),
    )
    mask_3d = Mask3D.unmasked(
        n_channels=image_plane_grid_3d.n_channels,
        shape_2d=image_plane_grid_3d.shape_2d,
        pixel_scales=image_plane_grid_3d.pixel_scales,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
        settings=settings,
    )

    flux_threshold = settings.get("reconstruction", {}).get("flux_threshold", 0.0)
    sb_input_units = settings.get("reconstruction", {}).get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    dataset_instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=source_grid_3d,
        z_step_kms=z_step_kms,
        sb_map=sb_map,
        flux_threshold=flux_threshold,
        sb_input_units=sb_input_units,
        **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
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
    )
    return analysis_instance


def _centre_bounds_from_priors(settings):
    priors = settings["priors"]
    c0 = priors["centre_0"]
    c1 = priors["centre_1"]
    return (
        float(c0["lower_limit"]),
        float(c0["upper_limit"]),
    ), (
        float(c1["lower_limit"]),
        float(c1["upper_limit"]),
    )


def optimize_source_centre(settings, analysis_instance, truth_params):
    """Hold truth kinematics fixed; maximize log-likelihood in source centre."""
    (c0_lo, c0_hi), (c1_lo, c1_hi) = _centre_bounds_from_priors(settings)
    truth_instance = truth_instance_from_settings(settings)

    def neg_log_likelihood(params):
        instance = truth_instance_from_settings(settings)
        instance.galaxies.source.centre = (float(params[0]), float(params[1]))
        try:
            return -float(
                analysis_instance.log_likelihood_function(instance=instance)
            )
        except Exception:
            return 1.0e30

    x0 = [float(truth_params["centre_0"]), float(truth_params["centre_1"])]
    bounds = [(c0_lo, c0_hi), (c1_lo, c1_hi)]
    result = scipy_optimize.minimize(
        neg_log_likelihood,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
    )
    best_centre = (float(result.x[0]), float(result.x[1]))
    best_instance = truth_instance_from_settings(settings)
    best_instance.galaxies.source.centre = best_centre
    best_model_data = analysis_instance.model_data_from_instance(
        instance=best_instance
    )
    truth_model_data = analysis_instance.model_data_from_instance(
        instance=truth_instance
    )
    return {
        "bounds": {"centre_0": (c0_lo, c0_hi), "centre_1": (c1_lo, c1_hi)},
        "truth_centre": (truth_params["centre_0"], truth_params["centre_1"]),
        "best_centre": best_centre,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
        "optimizer_message": str(result.message),
        "truth_stats": _visibility_fit_stats(analysis_instance, truth_model_data),
        "best_stats": _visibility_fit_stats(analysis_instance, best_model_data),
        "best_instance": best_instance,
        "best_model_data": best_model_data,
    }


def run_phase2_nautilus(settings, analysis_instance):
    """Full Nautilus phase 2 with truth kinematics and free centre only."""
    priors_cfg = priors_for_normalization_mode(
        priors_cfg=settings["priors"],
        mode=PIXELIZED,
        settings=settings,
    )
    model = af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(profiles.kinMSPixelized, priors_cfg)
        )
    )
    search = build_search_from_settings(settings["search"])
    return search.fit(model=model, analysis=analysis_instance)


def run_test(
    settings,
    *,
    phase1_only=False,
    run_nautilus=False,
    save_plots=True,
):
    truth_params = truth_parameters_from_settings(settings)
    phase2_cfg = settings.get("phase2", {})
    free_keys = tuple(phase2_cfg.get("free_parameters", ("centre_0", "centre_1")))
    if not phase2_cfg.get("kinematics_from_truth", True):
        print("  WARN — phase2.kinematics_from_truth is false; test assumes truth kinematics.")

    print("\n=== Phase-2 pixelized test setup ===\n")
    print(f"  model_name = {settings.get('model_name')}")
    print(f"  Truth kinematics (fixed in phase 2 except {list(free_keys)}):")
    for key in PIXELIZED_KINEMATIC_KEYS:
        marker = "  [FREE]" if key in free_keys else "  [fixed]"
        print(f"    {marker} {key} = {truth_params[key]!r}")

    out_root = _output_dir(settings)
    phase1_bundle = run_phase1(settings, output_dir=out_root if save_plots else None)

    if phase1_only:
        print(f"\n=== Phase 1 complete (plots under {out_root / 'phase1'}) ===\n")
        return {"phase1": phase1_bundle}

    print("\n=== Phase 2: truth kinematics + free source centre ===\n")
    print(
        f"  Lens centre (from phase 1) = "
        f"({phase1_bundle['lens_centre'][0]:.4f}, {phase1_bundle['lens_centre'][1]:.4f})"
    )

    analysis_instance = build_phase2_analysis(settings, phase1_bundle)

    centre_report = optimize_source_centre(
        settings, analysis_instance, truth_params
    )
    tc = centre_report["truth_centre"]
    bc = centre_report["best_centre"]
    truth_stats = centre_report["truth_stats"]
    best_stats = centre_report["best_stats"]

    print("  Centre optimization (scipy, other kinematics at truth):")
    (c0_lo, c0_hi) = centre_report["bounds"]["centre_0"]
    (c1_lo, c1_hi) = centre_report["bounds"]["centre_1"]
    print(f"    search centre_0 = [{c0_lo:.4f}, {c0_hi:.4f}] arcsec")
    print(f"    search centre_1 = [{c1_lo:.4f}, {c1_hi:.4f}] arcsec")
    print(f"    truth centre      = ({tc[0]:.4f}, {tc[1]:.4f})")
    print(f"    best-fit centre   = ({bc[0]:.4f}, {bc[1]:.4f})")
    print(
        f"    Δ centre (best − truth) = "
        f"({bc[0] - tc[0]:+.4f}, {bc[1] - tc[1]:+.4f}) arcsec"
    )
    print(f"    optimizer success = {centre_report['optimizer_success']}")
    print(f"    optimizer nfev    = {centre_report['optimizer_nfev']}")
    print("\n  At truth centre (kinematics fixed):")
    print(f"    chi²/N = {truth_stats['chi_squared_per_datum']:.6f}")
    print(f"    log L  = {truth_stats['log_likelihood']:.2f}")
    print("\n  At best-fit centre:")
    print(f"    chi²/N = {best_stats['chi_squared_per_datum']:.6f}")
    print(f"    log L  = {best_stats['log_likelihood']:.2f}")
    delta_ll = best_stats["log_likelihood"] - truth_stats["log_likelihood"]
    print(f"\n  Δ log L (best − truth) = {delta_ll:+.2f}")

    centre_dist = float(np.hypot(bc[0] - tc[0], bc[1] - tc[1]))
    if centre_dist < 0.05 and best_stats["chi_squared_per_datum"] < 1.2:
        print("\n  PASS — truth centre near optimum with pixelized SB + phase-1 lens.")
    elif best_stats["chi_squared_per_datum"] < truth_stats["chi_squared_per_datum"] - 1e-4:
        print(
            "\n  CENTRE OFFSET — a different source centre improves the UV fit; "
            "check phase-1 SB / lens centre / mock alignment."
        )
    else:
        print("\n  REVIEW — compare phase-1 SB and lens centre to mock expectations.")

    result = {
        "phase1": phase1_bundle,
        "centre_optimization": centre_report,
    }

    if run_nautilus:
        print("\n=== Phase 2: Nautilus (truth kinematics, free centre only) ===\n")
        nautilus_result = run_phase2_nautilus(settings, analysis_instance)
        result["nautilus"] = nautilus_result
        inst = nautilus_result.max_log_likelihood_instance
        centre = inst.galaxies.source.centre
        print(f"  Nautilus MAP centre = ({centre[0]:.4f}, {centre[1]:.4f})")
        if save_plots:
            plot_phase2_fit(
                analysis_instance=analysis_instance,
                result=nautilus_result,
                output_dir=out_root / "phase2_nautilus",
                settings=settings,
            )

    if save_plots:
        opt_dir = out_root / "phase2_optimized_centre"
        fit_dir = opt_dir / "fit_dataset"
        fit_dir.mkdir(parents=True, exist_ok=True)
        analysis_instance.visualizer.directory = str(fit_dir)
        analysis_instance.visualizer.visualize_data()
        analysis_instance.visualizer.visualize(
            model_data=centre_report["best_model_data"],
            during_analysis=False,
        )
        print(f"\n=== Plots saved under {out_root} ===\n")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run phase 1 only (SB reconstruction + free lens centre).",
    )
    parser.add_argument(
        "--run-nautilus",
        action="store_true",
        help="After scipy centre optimization, run full Nautilus phase 2.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.settings)

    run_test(
        settings,
        phase1_only=args.phase1_only,
        run_nautilus=args.run_nautilus,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
