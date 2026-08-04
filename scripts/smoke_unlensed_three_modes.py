#!/usr/bin/env python3
"""
Smoke-test all three KinMS normalization modes with ``lensing.enabled: false``.

1. Ensures the shared unlensed mock exists (regenerates if missing).
2. Mode 1 (parametric): truth-instance visibility likelihood.
3. Mode 2 (flux-from-phase1): phase-1 rectangular reconstruction + truth
   kinematics likelihood with phase-1 intensity.
4. Mode 3 (pixelized): phase-1 reconstruction + KinMSPixelized truth
   kinematics likelihood.

Does **not** run full nested sampling. Writes a JSON summary and dirty
data/model/residual plots under ``output/kinms_mock_unlensed_smoke/``.

Examples::

  python scripts/smoke_unlensed_three_modes.py
  python scripts/smoke_unlensed_three_modes.py --skip-generate
  python scripts/smoke_unlensed_three_modes.py --no-noise
  python scripts/smoke_unlensed_three_modes.py --modes parametric pixelized
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup

setup(__file__)

import matplotlib.pyplot as plt
import numpy as np

from scripts.generate_unlensed_mock_and_diagnose import generate_mock
from scripts.test_truth_model import _visibility_fit_stats
from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines import reconstruction
from src.pipelines.lens_model import (
    free_lens_centre_from_settings,
    lensing_enabled,
    validate_lensing_settings,
)
from src.pipelines.normalization import (
    PARAMETRIC,
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
    validate_normalization_settings,
)
from src.pipelines.pixelized_plots import plot_phase1_fit, save_cube, save_fit_triplet
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import autolens_utils, kinms_utils, plot_utils, spectral_utils

ROOT = Path(__file__).resolve().parents[1]

MODE_SETTINGS = {
    "parametric": "settings/runners/kinms_mock_unlensed_parametric.json",
    "parametric_flux": "settings/runners/kinms_mock_unlensed_parametric_flux.json",
    "pixelized": "settings/runners/kinms_mock_unlensed_pixelized.json",
}

DEFAULT_TEMPLATE_DATA = "data/kinms_mock_pixelized"
DEFAULT_TEMPLATE_UID = "kinms_mock"
DEFAULT_SUMMARY = "output/kinms_mock_unlensed_smoke/summary.json"
DEFAULT_PLOTS = "output/kinms_mock_unlensed_smoke/plots"


def _mock_visibilities_exist(settings):
    directory = Path(settings["data_directory"])
    uid = settings["uids"][0]
    width = settings["width"]
    patterns = settings["data_patterns"]
    name = patterns["visibilities"].format(uid=uid, width=width)
    path = directory / name
    if path.is_file():
        return True
    # dataprep patterns may include an extension; also accept stem.*
    stem = path.with_suffix("") if path.suffix else path
    return any(stem.parent.glob(stem.name + ".*"))


def _analysis_for_settings(settings, *, instance_obj, tracer):
    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    autolens_utils.resolve_image_plane_grid_in_settings(settings, uv_wavelengths)
    img_n, img_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=img_n,
        pixel_scale=img_scale,
        n_channels=len(frequencies),
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    mask_3d = Mask3D.unmasked(
        n_channels=image_grid_3d.n_channels,
        shape_2d=image_grid_3d.shape_2d,
        pixel_scales=image_grid_3d.pixel_scales,
    )
    masked_dataset = MaskedDataset(
        dataset=dataset, mask_3d=mask_3d, instance=instance_obj
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=masked_dataset.uv_wavelengths,
        mask_3d=masked_dataset.mask_3d,
        settings=settings,
    )
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )


def _extent_from_settings(settings):
    _, _, real_space_width = autolens_utils.source_grid_from_settings(settings)
    return autolens_utils.image_extent_arcsec(real_space_width)


def _save_kinematic_fit_plots(analysis, model_data, plots_dir, *, label, settings):
    """Dirty mom0 + channel residual mosaics for a kinematic forward model."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    z_step = float(analysis.masked_dataset.z_step_kms)
    extent = _extent_from_settings(settings)

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    dirty_model = autolens_utils.dirty_cube_from(
        visibilities=model_data,
        transformers=analysis.transformers,
    )
    data_mom0 = dirty_data.sum(axis=0) * z_step
    model_mom0 = dirty_model.sum(axis=0) * z_step
    resid_mom0 = data_mom0 - model_mom0

    dirty_sigma = autolens_utils.dirty_noise_cube_from(
        noise_map=analysis.masked_dataset.noise_map,
        transformers=analysis.transformers,
        n_realizations=8,
        seed=0,
    )
    mom0_sigma = autolens_utils.dirty_mom0_noise_from_channel_noise(
        dirty_sigma, z_step
    )

    save_fit_triplet(
        data_mom0,
        model_mom0,
        resid_mom0,
        plots_dir / "dirty_mom0_fit.png",
        titles=(
            f"{label}: dirty data (mom0)",
            f"{label}: dirty model (mom0)",
            f"{label}: residuals (mom0)",
        ),
        extent=extent,
        scale_mode="sigma",
        residual_sigma=mom0_sigma,
    )
    save_cube(
        dirty_data,
        plots_dir / "dirty_channels_data.png",
        ncols=8,
        extent=extent,
    )
    save_cube(
        dirty_model,
        plots_dir / "dirty_channels_model.png",
        ncols=8,
        extent=extent,
    )
    resid_cube = dirty_data - dirty_model
    lim = float(np.nanmax(np.abs(resid_cube))) or 1.0
    fig, _ = plot_utils.plot_cube(
        cube=resid_cube, ncols=8, vmin=-lim, vmax=lim, cmap="RdBu_r", extent=extent
    )
    fig.suptitle(f"{label}: channel residuals (data − model)", y=1.02)
    fig.savefig(plots_dir / "channel_residuals.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  plots -> {plots_dir}")
    return {
        "dirty_mom0_fit": str(plots_dir / "dirty_mom0_fit.png"),
        "channel_residuals": str(plots_dir / "channel_residuals.png"),
    }


def smoke_parametric(settings, *, plots_dir=None):
    validate_lensing_settings(settings)
    mode = validate_normalization_settings(settings)
    assert mode == PARAMETRIC
    assert not lensing_enabled(settings)
    assert not free_lens_centre_from_settings(settings)

    frequencies, _, _, _ = load_cube_data(settings)
    kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    instance_obj = kinms_utils.make_instance_from_grid(
        grid_3d=kinms_grid_3d,
        z_step_kms=z_step_kms,
        attach_grid=True,
        disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
    )
    analysis = _analysis_for_settings(
        settings,
        instance_obj=instance_obj,
        tracer=build_tracer(settings),
    )
    truth = truth_instance_from_settings(settings)
    model_data = analysis.model_data_from_instance(instance=truth)
    stats = _visibility_fit_stats(analysis, model_data)
    out = {
        "mode": mode,
        "mesh_type": None,
        "ok": True,
        "chi_squared_per_datum": stats["chi_squared_per_datum"],
        "log_likelihood": stats["log_likelihood"],
    }
    if plots_dir is not None:
        out["plots"] = _save_kinematic_fit_plots(
            analysis,
            model_data,
            plots_dir,
            label="parametric",
            settings=settings,
        )
    return out


def _smoke_two_phase(settings, *, expect_mode, plots_dir=None, mode_label="two_phase"):
    validate_lensing_settings(settings)
    mode = validate_normalization_settings(settings)
    assert mode == expect_mode
    assert not lensing_enabled(settings)
    assert settings["reconstruction"]["mesh_type"] == "rectangular_uniform"
    assert settings["reconstruction"]["regularization"]["type"] in {
        "constant",
        "adapt",
    }

    settings = copy.deepcopy(settings)
    settings["output_path"] = str(
        Path(settings["output_path"]).parent
        / (Path(settings["output_path"]).name.rstrip("/") + "_smoke")
    )
    settings["plot"] = False
    settings.setdefault("reconstruction", {}).setdefault("search", {})
    settings["reconstruction"]["search"]["maxiter"] = min(
        int(settings["reconstruction"]["search"].get("maxiter", 200)),
        50,
    )
    settings["reconstruction"]["search"]["number_of_cores"] = 1

    result = reconstruction.run_reconstruction(settings)
    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    n_pixels, pixel_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_plane_grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=len(frequencies),
    )
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
    flux_threshold = settings["reconstruction"].get("flux_threshold", 0.0)
    sb_input_units = settings["reconstruction"].get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    cloud_kwargs = kinms_utils.pixelized_instance_kwargs_from_settings(settings)

    if mode == PIXELIZED:
        dataset_instance = kinms_utils.make_pixelized_instance_from_grid(
            grid_3d=source_grid_3d,
            z_step_kms=z_step_kms,
            sb_map=sb_map,
            flux_threshold=flux_threshold,
            sb_input_units=sb_input_units,
            **cloud_kwargs,
        )
    else:
        total_flux = kinms_utils.kinms_intflux_from_sb_map(
            sb_map=sb_map,
            z_step_kms=z_step_kms,
            n_channels=source_grid_3d.n_channels,
            sb_input_units=sb_input_units,
        )
        dataset_instance = kinms_utils.make_instance_from_grid(
            grid_3d=source_grid_3d,
            z_step_kms=z_step_kms,
            attach_grid=True,
            disk_thick=cloud_kwargs["scale_height_arcsec"],
        )
        dataset_instance.int_flux = total_flux

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
    analysis = analysis_mod.Analysis(
        masked_dataset=MaskedDataset(
            dataset=dataset, mask_3d=mask_3d, instance=dataset_instance
        ),
        transformers=transformers,
        tracer=build_tracer(settings, centre=lens_centre),
        settings=settings,
    )
    truth = truth_instance_from_settings(settings)
    if mode == PARAMETRIC_FLUX_FROM_PHASE1:
        truth.galaxies.source.intensity = float(dataset_instance.int_flux)
    model_data = analysis.model_data_from_instance(instance=truth)
    stats = _visibility_fit_stats(analysis, model_data)
    out = {
        "mode": mode,
        "mesh_type": settings["reconstruction"]["mesh_type"],
        "ok": True,
        "phase1_sb_sum": float(np.sum(sb_map)),
        "lens_centre": [float(lens_centre[0]), float(lens_centre[1])],
        "chi_squared_per_datum": stats["chi_squared_per_datum"],
        "log_likelihood": stats["log_likelihood"],
    }
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        phase1_dir = plots_dir / "phase1"
        plot_phase1_fit(
            result=result,
            sb_map=sb_full,
            output_dir=phase1_dir,
            settings=settings,
            sb_map_extent=autolens_utils.image_extent_from_bounding_box(
                reconstruction.source_plane_bounding_box_from_result(result)
            ),
        )
        kin_plots = _save_kinematic_fit_plots(
            analysis,
            model_data,
            plots_dir / "phase2",
            label=mode_label,
            settings=settings,
        )
        out["plots"] = {
            "phase1": str(phase1_dir / "fit_triplet.png"),
            "phase1_sb_map": str(phase1_dir / "sb_map.png"),
            **{f"phase2_{k}": v for k, v in kin_plots.items()},
        }
    return out


def smoke_parametric_flux(settings, *, plots_dir=None):
    return _smoke_two_phase(
        settings,
        expect_mode=PARAMETRIC_FLUX_FROM_PHASE1,
        plots_dir=plots_dir,
        mode_label="parametric_flux",
    )


def smoke_pixelized(settings, *, plots_dir=None):
    return _smoke_two_phase(
        settings,
        expect_mode=PIXELIZED,
        plots_dir=plots_dir,
        mode_label="pixelized",
    )


SMOKE_FNS = {
    "parametric": smoke_parametric,
    "parametric_flux": smoke_parametric_flux,
    "pixelized": smoke_pixelized,
}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test unlensed three-mode KinMS pipelines."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_SETTINGS),
        default=sorted(MODE_SETTINGS),
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.set_defaults(add_noise=True)
    parser.add_argument(
        "--add-noise",
        action="store_true",
        dest="add_noise",
        help="Inject Gaussian visibility noise when generating (default)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_false",
        dest="add_noise",
        help=(
            "Noiseless mock. Diagnostic only — Autolens pixelizations "
            "struggle without a noise floor"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise injection")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for injected noise (and exported σ); e.g. 0.333 for ~3× quieter data",
    )
    parser.add_argument(
        "--pad-channels",
        type=int,
        default=None,
        help=(
            "Extra empty spectral channels at each end when regenerating the mock "
            "(overrides settings mock_pad_channels_each_side). Forces regenerate "
            "unless --skip-generate is also set."
        ),
    )
    parser.add_argument("--template-data", default=DEFAULT_TEMPLATE_DATA)
    parser.add_argument("--template-uid", default=DEFAULT_TEMPLATE_UID)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--plots-dir",
        default=DEFAULT_PLOTS,
        help="Root directory for data/model/residual plots (one subdir per mode)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing diagnostic plots",
    )
    args = parser.parse_args()

    parametric_settings = load_settings(str(ROOT / MODE_SETTINGS["parametric"]))
    validate_lensing_settings(parametric_settings)
    pad_channels = args.pad_channels
    if pad_channels is None:
        pad_channels = int(
            parametric_settings.get("mock_pad_channels_each_side", 0) or 0
        )

    need_generate = (not args.skip_generate) or (
        not _mock_visibilities_exist(parametric_settings)
    )
    # Changing the spectral axis requires a fresh mock even if files exist.
    if (not args.skip_generate) and pad_channels:
        need_generate = True
    if args.add_noise and args.skip_generate:
        print(
            "Warning: --add-noise with --skip-generate keeps the existing mock; "
            "noise is only applied when regenerating."
        )
    if (not args.add_noise) and (not args.skip_generate):
        print(
            "Warning: generating a noiseless mock; Autolens pixelizations "
            "typically need noise — omit --no-noise for production mocks"
        )
    if pad_channels and args.skip_generate:
        print(
            "Warning: --pad-channels with --skip-generate keeps the existing mock; "
            "padding is only applied when regenerating."
        )

    if need_generate:
        print("=== Ensuring shared unlensed mock exists ===")
        generate_mock(
            parametric_settings,
            template_data=args.template_data,
            template_uid=args.template_uid,
            add_noise=args.add_noise,
            seed=args.seed,
            pad_channels_each_side=pad_channels,
            noise_scale=args.noise_scale,
        )
    else:
        print("=== Using existing unlensed mock ===")

    plots_root = None if args.no_plots else Path(args.plots_dir)
    summary = {
        "modes": {},
        "lensing_enabled": False,
        "add_noise": bool(args.add_noise) and need_generate,
        "seed": int(args.seed) if (args.add_noise and need_generate) else None,
        "noise_scale": (
            float(args.noise_scale)
            if (args.add_noise and need_generate)
            else None
        ),
        "pad_channels_each_side": int(pad_channels) if need_generate else None,
        "plots_dir": None if plots_root is None else str(plots_root),
    }
    failed = False
    for mode_name in args.modes:
        settings_path = ROOT / MODE_SETTINGS[mode_name]
        print(f"\n=== Smoke: {mode_name} ({settings_path.name}) ===")
        settings = load_settings(str(settings_path))
        mode_plots = None if plots_root is None else plots_root / mode_name
        try:
            result = SMOKE_FNS[mode_name](settings, plots_dir=mode_plots)
            summary["modes"][mode_name] = result
            print(
                f"  ok  chi2/datum={result['chi_squared_per_datum']:.6g}  "
                f"lnL={result['log_likelihood']:.6g}"
            )
        except Exception as exc:
            failed = True
            summary["modes"][mode_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")
    if plots_root is not None:
        print(f"Plots under {plots_root}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
