"""Shared helpers for phase-1 lens + phase-2 truth-kinematics tests."""
from __future__ import annotations

import copy
from pathlib import Path

import autofit as af
import numpy as np
from scipy import optimize as scipy_optimize

from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines import reconstruction
from src.pipelines.normalization import normalization_mode_from_settings
from src.pipelines.normalization import priors_for_normalization_mode
from src.pipelines.pixelized_plots import plot_phase1_fit, plot_phase2_fit
from src.pipelines.priors import source_model_from_profile
from src.pipelines.runner import build_tracer, load_cube_data
from src.pipelines.search import build_search_from_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.pipelines.truth_parameters import kinematic_parameter_keys_for_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils


def visibility_fit_stats(analysis_instance, model_data):
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


def phase2_output_dir(settings, subdir=None):
    root = Path(settings.get("output_path", ".")) / "phase2_test"
    return root if subdir is None else root / subdir


def run_phase1_for_lens_centre(settings, output_dir=None):
    """Phase 1: pixelized SB reconstruction (lens centre free or fixed)."""
    settings = copy.deepcopy(settings)
    if output_dir is not None:
        settings["output_path"] = str(output_dir)
    settings.setdefault("reconstruction", {})
    fix_lens = settings["reconstruction"].get("fix_lens", True)
    if fix_lens:
        print(
            "  Phase-1 fix_lens=true — lens centre fixed to lens_mass_model (truth)."
        )
    else:
        print("  Phase-1 fix_lens=false — fitting lens centre.")

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    label = "fixed lens" if fix_lens else "free lens centre"
    print(f"\n=== Phase 1: pixelized reconstruction ({label}) ===\n")
    result = reconstruction.run_reconstruction(settings)

    frequencies, _, _, _ = load_cube_data(settings)
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
        phase1_result=result,
    )
    flux_snr_threshold = reconstruction.flux_snr_threshold_from_settings(settings)
    sb_map, lens_centre, sb_full, _noise = reconstruction.source_sb_for_phase2_flux(
        result=result,
        grid_2d=source_grid_3d.grid_2d,
        snr_threshold=flux_snr_threshold,
    )
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    sb_input_units = settings.get("reconstruction", {}).get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    phase1_total_flux = kinms_utils.kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=source_grid_3d.n_channels,
        sb_input_units=sb_input_units,
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
    print(f"  Phase-1 total flux for phase 2 = {phase1_total_flux:.6g} Jy km/s")

    if output_dir is not None:
        plot_phase1_fit(
            result=result,
            sb_map=sb_full,
            output_dir=Path(output_dir) / "phase1",
            settings=settings,
            sb_map_extent=autolens_utils.image_extent_from_bounding_box(
                reconstruction.source_plane_bounding_box_from_result(result)
            ),
        )

    return {
        "result": result,
        "sb_map": sb_map,
        "sb_map_full": sb_full,
        "lens_centre": lens_centre,
        "phase1_total_flux": phase1_total_flux,
        "flux_snr_threshold": flux_snr_threshold,
        "source_grid_3d": source_grid_3d,
        "frequencies": frequencies,
    }


def _load_uv_data(settings, frequencies):
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
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    return dataset, mask_3d, transformers, z_step_kms


def build_phase2_analysis_parametric(settings, phase1_bundle, *, use_phase1_flux=True):
    """
    Phase 2 analysis with parametric KinMS source and phase-1 lens centre.

    If ``use_phase1_flux`` is True (mode 2), ``intFlux`` is the phase-1 total
    flux. Otherwise (mode 1) intensity comes from the fitted source profile /
    truth ``intensity``.
    """
    frequencies = phase1_bundle["frequencies"]
    lens_centre = phase1_bundle["lens_centre"]
    source_grid_3d = phase1_bundle["source_grid_3d"]

    dataset, mask_3d, transformers, z_step_kms = _load_uv_data(settings, frequencies)
    dataset_instance = kinms_utils.make_instance_from_grid(
        grid_3d=source_grid_3d,
        z_step_kms=z_step_kms,
        attach_grid=True,
        disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
    )
    if use_phase1_flux:
        dataset_instance.int_flux = phase1_bundle["phase1_total_flux"]
    else:
        dataset_instance.int_flux = None
    tracer = build_tracer(settings, centre=lens_centre)
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
        instance=dataset_instance,
    )
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )


def build_phase2_analysis_pixelized(settings, phase1_bundle):
    """Phase 2 analysis with pixelized KinMS source and phase-1 lens centre."""
    frequencies = phase1_bundle["frequencies"]
    sb_map = phase1_bundle["sb_map"]
    lens_centre = phase1_bundle["lens_centre"]
    source_grid_3d = phase1_bundle["source_grid_3d"]

    dataset, mask_3d, transformers, z_step_kms = _load_uv_data(settings, frequencies)
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
    tracer = build_tracer(settings, centre=lens_centre)
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
        instance=dataset_instance,
    )
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )


def centre_bounds_from_priors(settings):
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
    """Hold truth source parameters fixed; maximize log-likelihood in centre only."""
    (c0_lo, c0_hi), (c1_lo, c1_hi) = centre_bounds_from_priors(settings)

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
    truth_instance = truth_instance_from_settings(settings)
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
        "truth_stats": visibility_fit_stats(analysis_instance, truth_model_data),
        "best_stats": visibility_fit_stats(analysis_instance, best_model_data),
        "best_instance": best_instance,
        "best_model_data": best_model_data,
    }


def print_centre_optimization_report(centre_report):
    tc = centre_report["truth_centre"]
    bc = centre_report["best_centre"]
    truth_stats = centre_report["truth_stats"]
    best_stats = centre_report["best_stats"]
    (c0_lo, c0_hi) = centre_report["bounds"]["centre_0"]
    (c1_lo, c1_hi) = centre_report["bounds"]["centre_1"]

    print("  Centre optimization (scipy, other parameters at truth):")
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
    print("\n  At truth centre:")
    print(f"    chi²/N = {truth_stats['chi_squared_per_datum']:.6f}")
    print(f"    log L  = {truth_stats['log_likelihood']:.2f}")
    print("\n  At best-fit centre:")
    print(f"    chi²/N = {best_stats['chi_squared_per_datum']:.6f}")
    print(f"    log L  = {best_stats['log_likelihood']:.2f}")
    delta_ll = best_stats["log_likelihood"] - truth_stats["log_likelihood"]
    print(f"\n  Δ log L (best − truth) = {delta_ll:+.2f}")


def print_truth_parameter_summary(settings, free_keys):
    from src.pipelines.truth_parameters import truth_parameters_from_settings

    truth_params = truth_parameters_from_settings(settings)
    print(f"  model_name = {settings.get('model_name')}")
    print(f"  Truth parameters (fixed in phase 2 except {list(free_keys)}):")
    for key in kinematic_parameter_keys_for_settings(settings):
        marker = "  [FREE]" if key in free_keys else "  [fixed]"
        print(f"    {marker} {key} = {truth_params[key]!r}")
    return truth_params


def run_phase2_nautilus(settings, analysis_instance, profile_cls):
    mode = normalization_mode_from_settings(settings)
    priors_cfg = priors_for_normalization_mode(
        priors_cfg=settings["priors"],
        mode=mode,
        settings=settings,
    )
    model = af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(profile_cls, priors_cfg)
        )
    )
    search = build_search_from_settings(settings["search"])
    return search.fit(model=model, analysis=analysis_instance)


def save_phase2_plots(analysis_instance, model_data, output_dir):
    output_dir = Path(output_dir)
    fit_dir = output_dir / "fit_dataset"
    fit_dir.mkdir(parents=True, exist_ok=True)
    analysis_instance.visualizer.directory = str(fit_dir)
    analysis_instance.visualizer.visualize_data()
    analysis_instance.visualizer.visualize(
        model_data=model_data,
        during_analysis=False,
    )
