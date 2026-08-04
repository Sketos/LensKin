#!/usr/bin/env python3
"""
Diagnose mode-2 (parametric SB + phase-1 intFlux) UV fit against data.

Builds the phase-1 lens + flux, then evaluates the parametric KinMS model at
truth kinematics (source centre fixed) and writes dirty-image diagnostics:

  - moment-0 / moment-1 data–model–residual triplets
  - per-channel dirty cubes (data, model, residual)
  - source-plane and lensed model mom0 (Jy/pixel; own color scales)
  - UV fit statistics

Note: lensed Jy/pixel mom0 must not share a color scale with dirty maps —
NUFFT dirty amplitudes are a different convention. Use ``dirty_mom0_*`` for
data−model residuals.

Example::

  python scripts/diagnose_mode2_fit.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --force-phase1
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

import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_source_kinematic_modes import _phase1_bundle
from scripts.test_truth_model import _visibility_fit_stats
from src.pipelines.normalization import PARAMETRIC_FLUX_FROM_PHASE1
from src.pipelines.phase2_test import build_phase2_analysis_parametric
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.pipelines.truth_parameters import truth_parameters_from_settings
from src.utils import analysis_utils, autolens_utils, plot_utils, spectral_utils


def _moments(cube, z_step_kms, z_centre=0.0):
    cube = np.asarray(cube, dtype=float)
    n_chan = cube.shape[0]
    velocities = (np.arange(n_chan) - 0.5 * n_chan) * float(z_step_kms) + float(z_centre)
    mom0 = np.sum(cube, axis=0) * float(z_step_kms)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom1 = np.sum(cube * velocities[:, None, None], axis=0) * float(z_step_kms) / mom0
    mom1 = np.where(np.isfinite(mom1), mom1, 0.0)
    return mom0, mom1


def _peak_yx(image, grid_2d):
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    i, j = np.unravel_index(np.nanargmax(image), image.shape)
    return float(coords[i, j, 0]), float(coords[i, j, 1])


def _save_channel_residuals(data_cube, model_cube, path, extent, ncols=8):
    residual = np.asarray(data_cube) - np.asarray(model_cube)
    limit = float(np.nanmax(np.abs(residual)))
    figure, axes = plot_utils.plot_cube(
        cube=residual,
        ncols=min(ncols, residual.shape[0]),
        extent=extent,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
    )
    figure.suptitle("Dirty channel residuals (data − model)", y=1.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def save_mode2_fit_diagnostics(
    analysis,
    instance,
    settings,
    output_dir,
    *,
    label="truth_kinematics",
    z_step_kms=None,
    source_grid_2d=None,
):
    """Write dirty + image-plane diagnostics for one mode-2 model instance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_step = float(
        z_step_kms
        if z_step_kms is not None
        else analysis.masked_dataset.z_step_kms
    )
    z_centre = float(getattr(instance.galaxies.source, "z_centre", 0.0))
    _, _, real_space_width = autolens_utils.image_plane_grid_from_settings(settings)
    extent = autolens_utils.image_extent_arcsec(real_space_width)
    image_grid = analysis.masked_dataset.grid_3d.grid_2d

    model_data = analysis.model_data_from_instance(instance=instance)
    stats = _visibility_fit_stats(analysis, model_data)

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    dirty_model = autolens_utils.dirty_cube_from(
        visibilities=model_data,
        transformers=analysis.transformers,
    )
    data_mom0, data_mom1 = _moments(dirty_data, z_step, z_centre=z_centre)
    model_mom0, model_mom1 = _moments(dirty_model, z_step, z_centre=z_centre)

    print(f"\n=== Mode-2 UV fit ({label}) ===\n")
    print(f"  flip_velocity_axis          = {settings.get('flip_velocity_axis', False)}")
    print(
        f"  flip_kinms_y_before_lensing = "
        f"{settings.get('flip_kinms_y_before_lensing', True)}"
    )
    print(f"  chi-squared / N = {stats['chi_squared_per_datum']:.6f}")
    print(f"  log L           = {stats['log_likelihood']:.2f}")
    dy, dx = _peak_yx(data_mom0, image_grid)
    my, mx = _peak_yx(model_mom0, image_grid)
    print(f"  dirty data  mom0 peak (y, x) = ({dy:+.4f}, {dx:+.4f})")
    print(f"  dirty model mom0 peak (y, x) = ({my:+.4f}, {mx:+.4f})")
    print(
        f"  Δ peak (model − data)        = ({my - dy:+.4f}, {mx - dx:+.4f}) arcsec"
    )
    print(f"  Plots -> {output_dir}")

    save_fit_triplet(
        data_mom0,
        model_mom0,
        data_mom0 - model_mom0,
        output_dir / "dirty_mom0_data_model_residual.png",
        titles=("Data dirty mom0", "Mode-2 model mom0", "Data − model"),
        extent=extent,
        scale_mode="residual",
    )
    save_fit_triplet(
        data_mom1,
        model_mom1,
        data_mom1 - model_mom1,
        output_dir / "dirty_mom1_data_model_residual.png",
        titles=("Data dirty mom1", "Mode-2 model mom1", "Data − model"),
        extent=extent,
        scale_mode="residual",
    )

    save_cube(dirty_data, output_dir / "dirty_channels_data.png", extent=extent)
    save_cube(dirty_model, output_dir / "dirty_channels_model.png", extent=extent)
    _save_channel_residuals(
        dirty_data,
        dirty_model,
        output_dir / "dirty_channels_residual.png",
        extent=extent,
    )

    # Also keep the visualizer-style fit_dataset tree.
    fit_dir = output_dir / "fit_dataset"
    fit_dir.mkdir(parents=True, exist_ok=True)
    analysis.visualizer.directory = str(fit_dir)
    analysis.visualizer.visualize_data()
    analysis.visualizer.visualize(model_data=model_data, during_analysis=False)

    # Source-plane / lensed model cubes (Jy/pixel; no UV). Do not compare these
    # on a shared color scale with dirty maps — NUFFT dirty images have a
    # different amplitude convention. Fit residuals belong in dirty_mom0_*.
    source_cube = analysis.model_cube_from_instance(instance=instance)
    if source_grid_2d is None and analysis.masked_dataset.instance is not None:
        source_grid_2d = getattr(
            analysis.masked_dataset.instance, "grid_3d", None
        )
        if source_grid_2d is not None:
            source_grid_2d = source_grid_2d.grid_2d
    flip_y = bool(getattr(analysis, "_flip_kinms_y", True))
    lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=source_cube,
        tracer=analysis.tracer,
        grid=image_grid,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=source_grid_2d,
        output_shape=analysis.masked_dataset.grid_3d.shape_2d,
        flip_kinms_y=flip_y,
    )
    src_mom0, _ = _moments(source_cube, z_step, z_centre=z_centre)
    lens_mom0, _ = _moments(lensed_cube, z_step, z_centre=z_centre)
    src_line = analysis_utils.integrated_line_flux_jy_kms(source_cube, z_step)
    lens_line = analysis_utils.integrated_line_flux_jy_kms(lensed_cube, z_step)
    print(f"  source-plane line flux = {src_line:.6g} Jy km/s")
    print(f"  lensed Jy/pixel line flux = {lens_line:.6g} Jy km/s")
    print(
        f"  dirty mom0 peak (data / model) = "
        f"{float(np.nanmax(data_mom0)):.4g} / {float(np.nanmax(model_mom0)):.4g}"
    )
    print(
        f"  lensed mom0 peak (Jy km/s/pix) = {float(np.nanmax(lens_mom0)):.4g} "
        f"(not comparable to dirty amplitude)"
    )
    if source_grid_2d is not None:
        coords = np.asarray(source_grid_2d).reshape(
            source_grid_2d.shape_native + (2,)
        )
        src_extent = (
            float(coords[0, :, 1].min()),
            float(coords[0, :, 1].max()),
            float(coords[:, 0, 0].min()),
            float(coords[:, 0, 0].max()),
        )
        flux_report = analysis_utils.flux_conservation_lensing_report(
            source_cube=source_cube,
            lensed_cube=lensed_cube,
            z_step_kms=z_step,
            source_grid_2d=source_grid_2d,
            image_grid_2d=image_grid,
        )
        print(
            f"  pixel area ratio (image/source) = "
            f"{flux_report['pixel_area_ratio_image_over_source']:.4g}"
        )
        print(
            f"  lensed/source line-flux ratio (magnification) = "
            f"{flux_report['line_flux_ratio']:.4g}"
        )
    else:
        src_extent = extent
    save_image(
        src_mom0,
        output_dir / "source_plane_model_mom0.png",
        title="Mode-2 source-plane mom0 (Jy km/s/pixel)",
        extent=src_extent,
    )
    save_image(
        lens_mom0,
        output_dir / "lensed_model_mom0.png",
        title="Mode-2 lensed mom0 (Jy km/s/pixel)",
        extent=extent,
    )
    # Morphology-only side-by-side: each panel uses its own scale so dirty vs
    # Jy/pixel units are not forced onto one colorbar.
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for axis, image, title, cmap in (
        (axes[0], data_mom0, "Dirty data mom0 (NUFFT)", "viridis"),
        (axes[1], lens_mom0, "Lensed model mom0 (Jy km/s/pixel)", "viridis"),
    ):
        im = axis.imshow(
            image,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap=cmap,
            extent=extent,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "Different units — use dirty_mom0_* for data−model residuals",
        y=1.02,
        fontsize=10,
    )
    side_path = output_dir / "dirty_data_mom0_and_lensed_jy_pixel_mom0.png"
    figure.savefig(side_path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    # Remove the misleading shared-scale triplet from earlier runs if present.
    stale = output_dir / "lensed_model_mom0_vs_dirty_data.png"
    if stale.is_file():
        stale.unlink()
    save_cube(source_cube, output_dir / "source_plane_channels.png", extent=src_extent)
    save_cube(lensed_cube, output_dir / "lensed_model_channels.png", extent=extent)

    # Side-by-side mid-channel dirty comparison
    mid = dirty_data.shape[0] // 2
    save_fit_triplet(
        dirty_data[mid],
        dirty_model[mid],
        dirty_data[mid] - dirty_model[mid],
        output_dir / f"dirty_channel_{mid:02d}_data_model_residual.png",
        titles=(f"Data ch {mid}", f"Model ch {mid}", "Residual"),
        extent=extent,
        scale_mode="residual",
    )

    return {
        "stats": stats,
        "output_dir": output_dir,
        "data_mom0": data_mom0,
        "model_mom0": model_mom0,
        "model_data": model_data,
    }


def run(settings_path, *, force_phase1=False, phase1_dir=None, plots_dir=None):
    settings = load_settings(settings_path)
    settings = copy.deepcopy(settings)
    settings["normalization_mode"] = PARAMETRIC_FLUX_FROM_PHASE1
    settings["model_name"] = "KinMS"

    out_root = Path(
        plots_dir
        if plots_dir is not None
        else Path(settings["output_path"]) / "mode2_fit_diagnostics"
    )
    cache_dir = Path(phase1_dir) if phase1_dir is not None else out_root
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Mode-2 (parametric_flux_from_phase1) fit diagnostics ===\n")
    print(f"  settings = {settings_path}")
    print(f"  output   = {out_root}")

    phase1_bundle = _phase1_bundle(
        settings, cache_dir, force_phase1=force_phase1
    )
    analysis = build_phase2_analysis_parametric(
        settings, phase1_bundle, use_phase1_flux=True
    )
    instance = truth_instance_from_settings(settings)
    truth = truth_parameters_from_settings(settings)

    print(f"  Phase-1 lens centre = {phase1_bundle['lens_centre']}")
    print(f"  Phase-1 total flux  = {phase1_bundle['phase1_total_flux']:.6g} Jy km/s")
    print(
        f"  Truth source centre (x, y) = "
        f"({truth['centre_0']}, {truth['centre_1']})"
    )
    print(
        f"  Truth kinematics: phi={truth['phi']}, inc={truth['inclination']}, "
        f"vmax={truth['maximum_velocity']}, sigma={truth['velocity_dispersion']}, "
        f"re={truth.get('effective_radius')}"
    )

    result = save_mode2_fit_diagnostics(
        analysis,
        instance,
        settings,
        out_root / "at_truth_centre",
        label="truth centre + truth kinematics + phase-1 flux",
        z_step_kms=spectral_utils.z_step_kms_from_data_frequencies(
            phase1_bundle["frequencies"]
        ),
        source_grid_2d=phase1_bundle["source_grid_3d"].grid_2d,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        default="settings/runners/kinms_mock_parametric_truth_kinematics.json",
    )
    parser.add_argument(
        "--force-phase1",
        action="store_true",
        help="Re-run phase 1 even if phase1_bundle.npz exists.",
    )
    parser.add_argument(
        "--phase1-dir",
        default=None,
        help="Directory for phase-1 cache (default: under output plots dir).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Output directory (default: <output_path>/mode2_fit_diagnostics).",
    )
    args = parser.parse_args()
    run(
        args.settings,
        force_phase1=args.force_phase1,
        phase1_dir=args.phase1_dir,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
