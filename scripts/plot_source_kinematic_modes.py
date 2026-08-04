#!/usr/bin/env python3
"""
Plot source-plane KinMS cubes for the three surface-brightness normalisation modes.

No lensing — compares the unlensed kinematic model only, including moment 0/1/2 maps.

Modes:
  1. parametric              — exponential disk + truth ``intensity``
  2. parametric_flux_from_phase1 — same parametric shape, ``intFlux`` from phase 1
  3. pixelized               — phase-1 SB map via ``inClouds`` / ``flux_clouds``

Optional UV centre test (``--run-phase2-centre``):
  Phase 1 — pixelized reconstruction. Lens centre follows
            ``reconstruction.fix_lens`` (``false`` = free lens centre).
  Phase 2 — for **all three** SB modes, fix truth kinematics/shape and
            optimize **source centre** only. Every mode uses the **phase-1
            lens centre**.

Examples:
  python scripts/plot_source_kinematic_modes.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json

  python scripts/plot_source_kinematic_modes.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --run-phase2-centre

  # Dedicated mode-2 centre test (same phase-1 / phase-2 free params):
  python scripts/test_phase2_parametric.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json
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

from src.pipelines.normalization import (
    PARAMETRIC,
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
)
from src.pipelines.phase2_test import (
    build_phase2_analysis_parametric,
    build_phase2_analysis_pixelized,
    optimize_source_centre,
    print_centre_optimization_report,
    print_truth_parameter_summary,
    run_phase1_for_lens_centre,
    save_phase2_plots,
)
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import load_settings
from src.pipelines.truth_model import (
    truth_instance_from_parameters,
    truth_instance_from_settings,
)
from src.pipelines.truth_parameters import truth_parameters_from_settings
from src.utils import autolens_utils, kinms_utils, plot_utils, spectral_utils


MODE_LABELS = {
    PARAMETRIC: "parametric",
    PARAMETRIC_FLUX_FROM_PHASE1: "parametric_flux_from_phase1",
    PIXELIZED: "pixelized",
}


def _phase1_bundle_path(output_dir):
    return Path(output_dir) / "phase1_bundle.npz"


def _save_phase1_bundle(path, bundle):
    np.savez(
        path,
        sb_map=bundle["sb_map"],
        phase1_total_flux=float(bundle["phase1_total_flux"]),
        lens_centre=np.asarray(bundle["lens_centre"], dtype=float),
        frequencies=np.asarray(bundle["frequencies"], dtype=float),
    )


def _load_phase1_bundle(path, settings):
    data = np.load(path)
    frequencies = np.asarray(data["frequencies"], dtype=float)
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
    )
    return {
        "sb_map": np.asarray(data["sb_map"], dtype=float),
        "phase1_total_flux": float(data["phase1_total_flux"]),
        "lens_centre": tuple(np.asarray(data["lens_centre"], dtype=float)),
        "source_grid_3d": source_grid_3d,
        "frequencies": frequencies,
    }


def _phase1_bundle(settings, cache_dir, *, force_phase1=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = _phase1_bundle_path(cache_dir)
    if bundle_path.is_file() and not force_phase1:
        print(f"\n=== Loading cached phase-1 bundle from {bundle_path} ===\n")
        return _load_phase1_bundle(bundle_path, settings)

    bundle = run_phase1_for_lens_centre(
        settings,
        output_dir=cache_dir / "phase1",
    )
    _save_phase1_bundle(bundle_path, bundle)
    print(f"  Cached phase-1 bundle -> {bundle_path}")
    return bundle


def _output_dir(settings, plots_dir=None):
    if plots_dir is not None:
        return Path(plots_dir)
    return Path(settings.get("output_path", ".")) / "source_kinematic_modes"


def _grid_extent(grid_3d):
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    y = coords[:, 0, 0]
    x = coords[0, :, 1]
    half_dy = abs(float(y[1] - y[0])) / 2.0 if len(y) > 1 else 0.0
    half_dx = abs(float(x[1] - x[0])) / 2.0 if len(x) > 1 else 0.0
    return (
        float(x.min() - half_dx),
        float(x.max() + half_dx),
        float(y.min() - half_dy),
        float(y.max() + half_dy),
    )


def _channel_velocities(n_channels, z_step_kms, z_centre=0.0):
    channel_indices = np.arange(n_channels, dtype=float)
    return (channel_indices - n_channels / 2.0) * z_step_kms + z_centre


def _moments_from_cube(cube, z_step_kms, z_centre=0.0, flux_threshold=0.0):
    cube = np.asarray(cube, dtype=float)
    velocities = _channel_velocities(cube.shape[0], z_step_kms, z_centre=z_centre)

    mom0 = np.sum(cube, axis=0) * z_step_kms
    with np.errstate(divide="ignore", invalid="ignore"):
        mom1 = np.sum(cube * velocities[:, None, None], axis=0) * z_step_kms / mom0
        mom2_sq = (
            np.sum(cube * (velocities[:, None, None] - mom1[None, :, :]) ** 2, axis=0)
            * z_step_kms
            / mom0
        )
        mom2 = np.sqrt(np.maximum(mom2_sq, 0.0))

    mask = mom0 <= flux_threshold
    mom1 = np.where(mask, np.nan, mom1)
    mom2 = np.where(mask, np.nan, mom2)
    return mom0, mom1, mom2


def _mom0(cube, z_step_kms, z_centre=0.0):
    return _moments_from_cube(cube, z_step_kms, z_centre=z_centre)[0]


def _save_moments_row(mom0, mom1, mom2, path, extent, title_prefix):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(
        mom0,
        origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
        cmap="viridis",
        extent=extent,
    )
    axes[0].set_title(f"{title_prefix} — moment 0 (Jy km/s/pix)")
    figure.colorbar(im0, ax=axes[0], fraction=0.046)

    vel_limit = np.nanpercentile(np.abs(mom1), 99.0)
    if not np.isfinite(vel_limit) or vel_limit <= 0:
        vel_limit = 1.0
    im1 = axes[1].imshow(
        mom1,
        origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
        cmap="RdBu_r",
        extent=extent,
        vmin=-vel_limit,
        vmax=vel_limit,
    )
    axes[1].set_title(f"{title_prefix} — moment 1 (km/s)")
    figure.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        mom2,
        origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
        cmap="magma",
        extent=extent,
        vmin=0.0,
        vmax=np.nanpercentile(mom2, 99.0) if np.any(np.isfinite(mom2)) else 1.0,
    )
    axes[2].set_title(f"{title_prefix} — moment 2 (km/s)")
    figure.colorbar(im2, ax=axes[2], fraction=0.046)

    for axis in axes:
        axis.set_xlabel("x (arcsec)")
        axis.set_ylabel("y (arcsec)")

    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def _save_moment_fit_triplet(reference, model, path, extent, titles, *, symmetric=False, cmap="viridis"):
    residual = model - reference
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    images = (reference, model, residual)

    if symmetric:
        limit = np.nanpercentile(np.abs(reference), 99.0)
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        vmins = (-limit, -limit, -limit)
        vmaxs = (limit, limit, limit)
    else:
        vmin = float(np.nanmin(reference))
        vmax = float(np.nanmax(reference))
        res_limit = np.nanpercentile(np.abs(residual), 99.0)
        if not np.isfinite(res_limit) or res_limit <= 0:
            res_limit = max(abs(vmin), abs(vmax), 1.0)
        vmins = (vmin, vmin, -res_limit)
        vmaxs = (vmax, vmax, res_limit)

    for axis, image, title, vmin, vmax in zip(axes, images, titles, vmins, vmaxs):
        im = axis.imshow(
            image,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("x (arcsec)")
        axis.set_ylabel("y (arcsec)")
        figure.colorbar(im, ax=axis, fraction=0.046)

    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def _save_moment_comparison(
    moment_maps,
    path,
    extent,
    titles,
    *,
    cmap="viridis",
    symmetric=False,
):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    images = (
        moment_maps[PARAMETRIC],
        moment_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        moment_maps[PIXELIZED] - moment_maps[PARAMETRIC],
    )
    if symmetric:
        limit = np.nanpercentile(np.abs(images[0]), 99.0)
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        vmins = (-limit, -limit, -limit)
        vmaxs = (limit, limit, limit)
    else:
        vmin = min(float(np.nanmin(image)) for image in images[:2])
        vmax = max(float(np.nanmax(image)) for image in images[:2])
        res_limit = np.nanpercentile(np.abs(images[2]), 99.0)
        if not np.isfinite(res_limit) or res_limit <= 0:
            res_limit = max(abs(vmin), abs(vmax), 1.0)
        vmins = (vmin, vmin, -res_limit)
        vmaxs = (vmax, vmax, res_limit)

    for axis, image, title, vmin, vmax in zip(axes, images, titles, vmins, vmaxs):
        im = axis.imshow(
            image,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("x (arcsec)")
        axis.set_ylabel("y (arcsec)")
        figure.colorbar(im, ax=axis, fraction=0.046)

    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def _line_flux(cube, z_step_kms):
    return float(np.sum(cube) * z_step_kms)


def _peak_yx(image, grid_3d):
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    i, j = np.unravel_index(np.nanargmax(image), image.shape)
    return float(coords[i, j, 0]), float(coords[i, j, 1])


def _source_grid(settings, frequencies, phase1_bundle=None):
    if phase1_bundle is not None:
        return phase1_bundle["source_grid_3d"]
    return autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )


def _truth_profiles(settings):
    params = truth_parameters_from_settings(settings)
    parametric = truth_instance_from_parameters(
        params=params, model_name="KinMS"
    ).galaxies.source
    pixelized = truth_instance_from_parameters(
        params=params, model_name="KinMSPixelized"
    ).galaxies.source
    return params, parametric, pixelized


def _cube_parametric(profile, grid_3d, z_step_kms, int_flux=None, disk_thick=0.0):
    instance = kinms_utils.make_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms,
        attach_grid=True,
        disk_thick=disk_thick,
    )
    if int_flux is not None:
        instance.int_flux = float(int_flux)
    return profile.profile_cube_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms,
        instance=instance,
    )


def _cube_pixelized(profile, grid_3d, z_step_kms, sb_map, settings):
    flux_threshold = settings.get("reconstruction", {}).get("flux_threshold", 0.0)
    sb_input_units = settings.get("reconstruction", {}).get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms,
        sb_map=sb_map,
        flux_threshold=flux_threshold,
        sb_input_units=sb_input_units,
        **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
    )
    return profile.profile_cube_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms,
        instance=instance,
    )


def build_source_cubes(settings, phase1_bundle):
    """Return dict mode -> source cube (no lensing)."""
    frequencies = phase1_bundle["frequencies"]
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    grid_3d = _source_grid(settings, frequencies, phase1_bundle)
    params, param_profile, pix_profile = _truth_profiles(settings)

    phase1_flux = phase1_bundle["phase1_total_flux"]
    sb_map = phase1_bundle["sb_map"]
    disk_thick = kinms_utils.disk_scale_height_arcsec_from_settings(settings)

    cubes = {
        PARAMETRIC: _cube_parametric(
            param_profile,
            grid_3d,
            z_step_kms,
            int_flux=params.get("intensity"),
            disk_thick=disk_thick,
        ),
        PARAMETRIC_FLUX_FROM_PHASE1: _cube_parametric(
            param_profile,
            grid_3d,
            z_step_kms,
            int_flux=phase1_flux,
            disk_thick=disk_thick,
        ),
        PIXELIZED: _cube_pixelized(
            pix_profile,
            grid_3d,
            z_step_kms,
            sb_map=sb_map,
            settings=settings,
        ),
    }
    return cubes, grid_3d, z_step_kms, params, phase1_flux


def _print_cube_summary(
    mode,
    cube,
    grid_3d,
    z_step_kms,
    reference_mom0=None,
    z_centre=0.0,
):
    mom0, mom1, mom2 = _moments_from_cube(cube, z_step_kms, z_centre=z_centre)
    peak_y, peak_x = _peak_yx(mom0, grid_3d)
    line_flux = _line_flux(cube, z_step_kms)
    print(f"  {MODE_LABELS[mode]}:")
    print(f"    cube shape           = {cube.shape}")
    print(f"    line flux            = {line_flux:.6g} Jy km/s")
    print(f"    mom0 peak (y, x)     = ({peak_y:+.4f}, {peak_x:+.4f}) arcsec")
    print(
        f"    mom1 median          = {float(np.nanmedian(mom1)):+.4f} km/s "
        f"(peak channel offset)"
    )
    print(f"    mom2 median          = {float(np.nanmedian(mom2)):.4f} km/s")
    if reference_mom0 is not None:
        residual = mom0 - reference_mom0
        ref_peak = float(np.nanmax(reference_mom0))
        rel_rms = float(np.sqrt(np.mean(residual**2)) / ref_peak) if ref_peak > 0 else float("nan")
        print(f"    mom0 residual RMS    = {float(np.sqrt(np.mean(residual**2))):.6g} Jy km/s/pix")
        print(f"    mom0 fractional RMS  = {rel_rms:.4f} vs parametric")


def plot_mode_comparison(cubes, grid_3d, z_step_kms, output_dir, z_centre=0.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = _grid_extent(grid_3d)
    moments = {
        mode: _moments_from_cube(cube, z_step_kms, z_centre=z_centre)
        for mode, cube in cubes.items()
    }
    mom0_maps = {mode: maps[0] for mode, maps in moments.items()}
    mom1_maps = {mode: maps[1] for mode, maps in moments.items()}
    mom2_maps = {mode: maps[2] for mode, maps in moments.items()}

    ref_mode = PARAMETRIC
    ref_mom0 = mom0_maps[ref_mode]
    ref_mom1 = mom1_maps[ref_mode]
    ref_mom2 = mom2_maps[ref_mode]

    # Per-mode moment 0/1/2 rows
    for mode in (PARAMETRIC, PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED):
        mom0, mom1, mom2 = moments[mode]
        _save_moments_row(
            mom0,
            mom1,
            mom2,
            output_dir / f"moments_{MODE_LABELS[mode]}.png",
            extent,
            MODE_LABELS[mode],
        )
        save_image(
            mom0,
            output_dir / f"mom0_{MODE_LABELS[mode]}.png",
            title=f"moment 0 — {MODE_LABELS[mode]}",
            extent=extent,
        )
        save_image(
            mom1,
            output_dir / f"mom1_{MODE_LABELS[mode]}.png",
            title=f"moment 1 — {MODE_LABELS[mode]}",
            cmap="RdBu_r",
            extent=extent,
        )
        save_image(
            mom2,
            output_dir / f"mom2_{MODE_LABELS[mode]}.png",
            title=f"moment 2 — {MODE_LABELS[mode]}",
            cmap="magma",
            extent=extent,
        )

    # Moment-0 triplet
    save_fit_triplet(
        data=mom0_maps[PARAMETRIC],
        model=mom0_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        residuals=mom0_maps[PIXELIZED] - ref_mom0,
        path=output_dir / "mom0_parametric_flux_pixelized.png",
        titles=(
            "parametric (truth intensity)",
            "parametric_flux_from_phase1",
            "pixelized − parametric",
        ),
        extent=extent,
        scale_mode="data",
    )

    # All three mom0 maps with shared scale
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = max(float(np.nanmax(m)) for m in mom0_maps.values())
    vmin = min(float(np.nanmin(m)) for m in mom0_maps.values())
    for ax, mode in zip(axes, (PARAMETRIC, PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED)):
        im = ax.imshow(
            mom0_maps[mode],
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap="viridis",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("x (arcsec)")
        ax.set_ylabel("y (arcsec)")
        figure.colorbar(im, ax=ax, fraction=0.046)
    figure.savefig(output_dir / "mom0_all_modes_shared_scale.png", bbox_inches="tight", dpi=150)
    plt.close(figure)

    # Residuals vs parametric reference
    save_fit_triplet(
        data=ref_mom0,
        model=mom0_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        residuals=mom0_maps[PARAMETRIC_FLUX_FROM_PHASE1] - ref_mom0,
        path=output_dir / "mom0_flux_from_phase1_vs_parametric.png",
        titles=("parametric", "flux_from_phase1", "flux_from_phase1 − parametric"),
        extent=extent,
        scale_mode="data",
    )
    save_fit_triplet(
        data=ref_mom0,
        model=mom0_maps[PIXELIZED],
        residuals=mom0_maps[PIXELIZED] - ref_mom0,
        path=output_dir / "mom0_pixelized_vs_parametric.png",
        titles=("parametric", "pixelized", "pixelized − parametric"),
        extent=extent,
        scale_mode="data",
    )

    _save_moment_comparison(
        mom1_maps,
        output_dir / "mom1_parametric_flux_pixelized.png",
        extent,
        (
            "parametric (truth intensity)",
            "parametric_flux_from_phase1",
            "pixelized − parametric",
        ),
        cmap="RdBu_r",
        symmetric=True,
    )
    _save_moment_fit_triplet(
        ref_mom1,
        mom1_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        output_dir / "mom1_flux_from_phase1_vs_parametric.png",
        extent,
        ("parametric", "flux_from_phase1", "flux_from_phase1 − parametric"),
        symmetric=True,
        cmap="RdBu_r",
    )
    _save_moment_fit_triplet(
        ref_mom1,
        mom1_maps[PIXELIZED],
        output_dir / "mom1_pixelized_vs_parametric.png",
        extent,
        ("parametric", "pixelized", "pixelized − parametric"),
        symmetric=True,
        cmap="RdBu_r",
    )

    _save_moment_comparison(
        mom2_maps,
        output_dir / "mom2_parametric_flux_pixelized.png",
        extent,
        (
            "parametric (truth intensity)",
            "parametric_flux_from_phase1",
            "pixelized − parametric",
        ),
        cmap="magma",
        symmetric=False,
    )
    save_fit_triplet(
        data=ref_mom2,
        model=mom2_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        residuals=mom2_maps[PARAMETRIC_FLUX_FROM_PHASE1] - ref_mom2,
        path=output_dir / "mom2_flux_from_phase1_vs_parametric.png",
        titles=("parametric", "flux_from_phase1", "flux_from_phase1 − parametric"),
        extent=extent,
        scale_mode="data",
    )
    save_fit_triplet(
        data=ref_mom2,
        model=mom2_maps[PIXELIZED],
        residuals=mom2_maps[PIXELIZED] - ref_mom2,
        path=output_dir / "mom2_pixelized_vs_parametric.png",
        titles=("parametric", "pixelized", "pixelized − parametric"),
        extent=extent,
        scale_mode="data",
    )

    # Central channel slice
    mid = cubes[PARAMETRIC].shape[0] // 2
    channel_maps = {mode: cube[mid] for mode, cube in cubes.items()}
    save_fit_triplet(
        data=channel_maps[PARAMETRIC],
        model=channel_maps[PARAMETRIC_FLUX_FROM_PHASE1],
        residuals=channel_maps[PIXELIZED] - channel_maps[PARAMETRIC],
        path=output_dir / f"channel_{mid}_comparison.png",
        titles=(
            f"parametric ch {mid}",
            f"flux_from_phase1 ch {mid}",
            f"pixelized − parametric ch {mid}",
        ),
        extent=extent,
        scale_mode="data",
    )

    # Position–velocity: slice through map centre along x
    ny, nx = grid_3d.shape_2d
    cy, cx = ny // 2, nx // 2
    coords = np.asarray(grid_3d.grid_2d).reshape(grid_3d.shape_2d + (2,))
    x_axis = coords[cy, :, 1]
    v_axis = _channel_velocities(cubes[PARAMETRIC].shape[0], z_step_kms, z_centre=z_centre)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mode in zip(axes, (PARAMETRIC, PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED)):
        pv = cubes[mode][:, cy, :]
        im = ax.imshow(
            pv,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            aspect="auto",
            cmap="jet",
            extent=(float(x_axis.min()), float(x_axis.max()), float(v_axis.min()), float(v_axis.max())),
        )
        ax.set_title(f"PV along y index {cy} — {MODE_LABELS[mode]}")
        ax.set_xlabel("x (arcsec)")
        ax.set_ylabel("v (km/s)")
        figure.colorbar(im, ax=ax, fraction=0.046)
    figure.savefig(output_dir / "pv_x_slice_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(figure)

    # Per-mode channel montages
    for mode, cube in cubes.items():
        save_cube(
            cube,
            output_dir / f"cube_{MODE_LABELS[mode]}.png",
            ncols=min(8, cube.shape[0]),
            extent=extent,
        )

    return output_dir


def run_mode_phase2_centre_test(settings, phase1_bundle, output_dir, mode):
    """
    UV source-centre test for one SB normalization mode.

    Phase-1 lens centre from ``phase1_bundle`` is used for all modes.
    """
    settings = copy.deepcopy(settings)
    settings["normalization_mode"] = mode
    phase2_cfg = settings.setdefault("phase2", {})
    phase2_cfg.setdefault("kinematics_from_truth", True)
    phase2_cfg.setdefault("free_parameters", ["centre_0", "centre_1"])
    free_keys = tuple(phase2_cfg["free_parameters"])

    label = MODE_LABELS[mode]
    print(f"\n=== Phase-2 centre test — {label} (UV fit) ===\n")
    print(
        f"  Phase-1 lens centre = "
        f"({phase1_bundle['lens_centre'][0]:.4f}, {phase1_bundle['lens_centre'][1]:.4f})"
    )

    if mode == PIXELIZED:
        settings["model_name"] = "KinMSPixelized"
        analysis_instance = build_phase2_analysis_pixelized(settings, phase1_bundle)
        print("  Source model = KinMSPixelized (phase-1 SB map)")
    elif mode == PARAMETRIC_FLUX_FROM_PHASE1:
        settings["model_name"] = "KinMS"
        analysis_instance = build_phase2_analysis_parametric(
            settings, phase1_bundle, use_phase1_flux=True
        )
        print(
            f"  Source model = KinMS parametric, "
            f"intFlux from phase 1 = {phase1_bundle['phase1_total_flux']:.6g} Jy km/s"
        )
    elif mode == PARAMETRIC:
        settings["model_name"] = "KinMS"
        analysis_instance = build_phase2_analysis_parametric(
            settings, phase1_bundle, use_phase1_flux=False
        )
        truth_intensity = truth_parameters_from_settings(settings).get("intensity")
        print(
            f"  Source model = KinMS parametric, "
            f"truth intensity = {truth_intensity!r} Jy km/s"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    truth_params = print_truth_parameter_summary(settings, free_keys)
    centre_report = optimize_source_centre(
        settings, analysis_instance, truth_params
    )
    print_centre_optimization_report(centre_report)

    phase2_dir = Path(output_dir) / f"phase2_{label}_optimized_centre"
    save_phase2_plots(
        analysis_instance,
        centre_report["best_model_data"],
        phase2_dir,
    )
    print(f"\n  Phase-2 plots -> {phase2_dir}")
    return centre_report


def run_all_modes_phase2_centre_test(settings, phase1_bundle, output_dir):
    """Run the UV source-centre test for parametric, mode-2, and pixelized."""
    reports = {}
    for mode in (PARAMETRIC, PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED):
        reports[mode] = run_mode_phase2_centre_test(
            settings, phase1_bundle, output_dir, mode
        )
    return reports


def run_mode2_phase2_centre_test(settings, phase1_bundle, output_dir):
    """Backward-compatible alias for the mode-2-only UV centre test."""
    return run_mode_phase2_centre_test(
        settings, phase1_bundle, output_dir, PARAMETRIC_FLUX_FROM_PHASE1
    )


def run(
    settings,
    *,
    output_dir=None,
    force_phase1=False,
    phase1_dir=None,
    run_phase2_centre=False,
):
    settings = copy.deepcopy(settings)
    if "reconstruction" not in settings:
        raise ValueError(
            "Settings must include a 'reconstruction' block for phase-1 SB "
            "(required for parametric_flux_from_phase1 and pixelized modes)."
        )
    if "truth_parameters" not in settings and "debug_instance_vector" not in settings:
        raise ValueError(
            "Settings must include 'truth_parameters' or 'debug_instance_vector'."
        )

    plot_dir = _output_dir(settings, plots_dir=output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    phase1_cache_dir = Path(phase1_dir) if phase1_dir is not None else plot_dir

    print("\n=== Source kinematic model comparison (no lensing) ===\n")
    truth = truth_instance_from_settings(settings)
    centre = truth.galaxies.source.centre
    print(f"  Truth source centre (x, y) = ({centre[0]:.4f}, {centre[1]:.4f})")
    print(f"  Truth phi / inc            = {truth.galaxies.source.phi:.1f}° / {truth.galaxies.source.inclination:.1f}°")
    print(
        f"  Phase-1 fix_lens           = "
        f"{settings.get('reconstruction', {}).get('fix_lens', True)!r} "
        f"(false => free lens centre)"
    )

    phase1_bundle = _phase1_bundle(
        settings,
        phase1_cache_dir,
        force_phase1=force_phase1,
    )

    cubes, grid_3d, z_step_kms, params, phase1_flux = build_source_cubes(
        settings, phase1_bundle
    )

    print("\n=== Source cube summary ===\n")
    print(f"  Grid = {grid_3d.n_pixels}², z_step = {z_step_kms:.4f} km/s")
    print(f"  Truth intensity            = {params.get('intensity')!r} Jy km/s")
    print(f"  Phase-1 total flux         = {phase1_flux:.6g} Jy km/s")
    print(f"  Phase-1 / truth intensity  = {phase1_flux / params['intensity']:.4g}")
    print(f"  Truth effective_radius     = {params.get('effective_radius')!r} arcsec")

    ref_mom0 = _mom0(cubes[PARAMETRIC], z_step_kms, z_centre=params.get("z_centre", 0.0))
    for mode in (PARAMETRIC, PARAMETRIC_FLUX_FROM_PHASE1, PIXELIZED):
        _print_cube_summary(
            mode,
            cubes[mode],
            grid_3d,
            z_step_kms,
            reference_mom0=ref_mom0,
            z_centre=params.get("z_centre", 0.0),
        )

    save_image(
        phase1_bundle["sb_map"],
        plot_dir / "phase1_sb_map.png",
        title="phase-1 SB map (Jy/pixel/channel)",
        extent=_grid_extent(grid_3d),
    )

    plot_dir = plot_mode_comparison(
        cubes,
        grid_3d,
        z_step_kms,
        plot_dir,
        z_centre=params.get("z_centre", 0.0),
    )
    print(f"\n=== Plots saved to {plot_dir} ===\n")
    for path in sorted(plot_dir.glob("*.png")):
        print(f"  {path.relative_to(plot_dir)}")

    result = {
        "cubes": cubes,
        "grid_3d": grid_3d,
        "z_step_kms": z_step_kms,
        "phase1_bundle": phase1_bundle,
        "output_dir": plot_dir,
    }

    if run_phase2_centre:
        result["centre_optimization"] = run_all_modes_phase2_centre_test(
            settings,
            phase1_bundle,
            plot_dir,
        )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="settings/runners/kinms_mock_parametric_truth_kinematics.json",
        help="Runner settings JSON with truth_parameters and reconstruction block.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison plots (default: <output_path>/source_kinematic_modes).",
    )
    parser.add_argument(
        "--force-phase1",
        action="store_true",
        help="Re-run phase-1 reconstruction even if phase1_bundle.npz exists.",
    )
    parser.add_argument(
        "--fix-lens",
        action="store_true",
        default=None,
        help="Force phase-1 lens centre to lens_mass_model (truth). "
        "Overrides reconstruction.fix_lens in settings.",
    )
    parser.add_argument(
        "--free-lens",
        action="store_true",
        default=None,
        help="Allow phase-1 to fit the lens centre. "
        "Overrides reconstruction.fix_lens in settings.",
    )
    parser.add_argument(
        "--phase1-dir",
        default=None,
        help="Directory for phase-1 outputs and phase1_bundle.npz cache.",
    )
    parser.add_argument(
        "--run-phase2-centre",
        action="store_true",
        help=(
            "After source-plane plots, run UV source-centre tests for all three "
            "SB modes, each using the phase-1 lens centre."
        ),
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    if args.fix_lens and args.free_lens:
        raise SystemExit("Pass only one of --fix-lens or --free-lens.")
    if args.fix_lens:
        settings.setdefault("reconstruction", {})["fix_lens"] = True
    elif args.free_lens:
        settings.setdefault("reconstruction", {})["fix_lens"] = False

    run(
        settings,
        output_dir=args.output_dir,
        force_phase1=args.force_phase1,
        phase1_dir=args.phase1_dir,
        run_phase2_centre=args.run_phase2_centre,
    )


if __name__ == "__main__":
    main()
