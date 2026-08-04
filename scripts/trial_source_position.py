#!/usr/bin/env python3
"""
Trial phaseCent signs and lensing axis flips against data peak / UV chi-squared.

Use when lensed dirty maps look offset but conventions audits pass.

Examples:
  python scripts/trial_source_position.py \\
    --settings settings/runners/kinms_mock_parametric_mock40.json

  # Settings with normalization_mode needing phase 1 (e.g. parametric_flux_from_phase1):
  python scripts/trial_source_position.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --run-phase1 --lens-centre phase1
"""
from __future__ import annotations

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
from scipy import interpolate

from scripts.test_truth_model import _build_pipeline, _visibility_fit_stats
from src.analysis.analysis import model_data_from_cube_and_transformers
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils


PHASE_CENT_VARIANTS = {
    "current [-y, x]": lambda c: [-c[1], c[0]],
    "[-x, -y]": lambda c: [-c[0], -c[1]],
    "[x, y]": lambda c: [c[0], c[1]],
    "[-x, y]": lambda c: [-c[0], c[1]],
    "[x, -y]": lambda c: [c[0], -c[1]],
}

SRC_FLIPS = ("none", "y", "x", "yx")
POST_FLIPS = ("none", "xmirror", "yflip", "yx")


def _peak_yx(mom0, grid_2d):
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    i, j = np.unravel_index(np.nanargmax(mom0), mom0.shape)
    return float(coords[i, j, 0]), float(coords[i, j, 1])


def _apply_2d_flip(image, mode):
    if mode == "y":
        return image[::-1, :]
    if mode == "x":
        return image[:, ::-1]
    if mode == "yx":
        return image[::-1, ::-1]
    return image


def _source_cube(instance, kinms_instance, phase_cent):
    prof = instance.galaxies.source
    x = kinms_instance.x
    sbprof = np.exp(-x / prof.effective_radius)
    velprof = np.hypot(
        (2.0 * prof.maximum_velocity / np.pi)
        * np.arctan(x / prof.turnover_radius),
        prof.vmax_black_hole / np.sqrt(x),
    )
    cube = kinms_instance.obj.model_cube(
        inc=prof.inclination,
        posAng=prof.phi,
        intFlux=prof.intensity,
        gasSigma=prof.velocity_dispersion,
        sbProf=sbprof,
        velProf=velprof,
        sbRad=x,
        velRad=x,
        inClouds=np.zeros((0, 3)),
        flux_clouds=None,
        phaseCent=list(phase_cent),
        vOffset=prof.z_centre,
    )
    # KinMS returns (x, y, v); autolens grids / ray-tracing expect (v, y, x).
    cube = cube.transpose(2, 1, 0)
    return cube


def _lens_manual(source_cube, tracer, source_grid_2d, image_grid_2d, src_flip, post_flip):
    traced = np.asarray(tracer.traced_grid_2d_list_from(grid=image_grid_2d)[1])
    shape = source_grid_2d.shape_native
    coords = np.asarray(source_grid_2d).reshape(shape + (2,))
    y_interp = coords[:, 0, 0]
    x_interp = coords[0, :, 1]
    omega_src = analysis_utils.pixel_area_arcsec2_from_grid_2d(source_grid_2d)
    omega_img = analysis_utils.pixel_area_arcsec2_from_grid_2d(image_grid_2d)
    scale = omega_img / omega_src if omega_src > 0 else 1.0
    out_shape = image_grid_2d.shape_native
    lensed = np.zeros((source_cube.shape[0],) + out_shape)
    for i, image in enumerate(source_cube):
        img = analysis_utils.resample_image_to_shape(image, shape)
        img = _apply_2d_flip(np.asarray(img), src_flip)
        interp = interpolate.RegularGridInterpolator(
            points=(y_interp, x_interp),
            values=img,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        plane = interp(traced).reshape(out_shape) * scale
        lensed[i] = _apply_2d_flip(plane, post_flip)
    return lensed


def run_trials(
    settings_path,
    top_n=15,
    *,
    run_phase1=False,
    lens_centre_mode="fixed",
):
    from src.pipelines.runner import load_settings

    settings = load_settings(settings_path)
    pipeline = _build_pipeline(
        settings,
        run_phase1=run_phase1,
        lens_centre_mode=lens_centre_mode,
    )
    analysis = pipeline["analysis"]
    image_grid = pipeline["image_grid_3d"].grid_2d
    source_grid = pipeline["kinms_grid_3d"].grid_2d
    z_step = pipeline["z_step_kms"]

    instance = truth_instance_from_settings(settings)
    kinms_instance = analysis.masked_dataset.instance
    centre = instance.galaxies.source.centre
    flip_kinms_y = bool(settings.get("flip_kinms_y_before_lensing", True))

    dirty = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    data_mom0 = dirty.sum(axis=0) * z_step
    data_peak = _peak_yx(data_mom0, image_grid)

    print("\n=== Source position trial ===\n")
    print(f"  settings = {settings_path}")
    print(f"  run_phase1 = {run_phase1}")
    print(f"  lens_centre_mode = {lens_centre_mode}")
    print(f"  lens centre = {pipeline['lens_centre']}")
    print(f"  flip_kinms_y_before_lensing = {flip_kinms_y}")
    print(f"  centre tuple (centre_0, centre_1) = (x, y) = {centre}")
    print(f"  data mom0 peak (y, x) = {data_peak}")

    # Production path (current analysis_utils)
    sc_prod = analysis.model_cube_from_instance(instance=instance)
    lc_prod = analysis_utils.lensed_cube_from_tracer(
        cube=sc_prod,
        tracer=analysis.tracer,
        grid=image_grid,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=source_grid,
        output_shape=image_grid.shape_native,
        flip_kinms_y=flip_kinms_y,
    )
    md_prod = analysis.model_data_from_instance(instance=instance)
    st_prod = _visibility_fit_stats(analysis, md_prod)
    prod_peak = _peak_yx(lc_prod.sum(axis=0) * z_step, image_grid)
    src_peak = _peak_yx(sc_prod.sum(axis=0) * z_step, source_grid)
    print(f"  production src peak (y, x) = {src_peak}")
    print(f"  production lens peak (y, x) = {prod_peak}")
    print(f"  production chi-squared / N = {st_prod['chi_squared_per_datum']:.6f}")

    rows = []
    for phase_name, phase_fn in PHASE_CENT_VARIANTS.items():
        phase_cent = phase_fn(centre)
        source_cube = _source_cube(instance, kinms_instance, phase_cent)
        for src_flip in SRC_FLIPS:
            for post_flip in POST_FLIPS:
                lensed = _lens_manual(
                    source_cube,
                    analysis.tracer,
                    source_grid,
                    image_grid,
                    src_flip,
                    post_flip,
                )
                model_data = model_data_from_cube_and_transformers(
                    cube=lensed,
                    transformers=analysis.transformers,
                    shape=analysis.masked_dataset.data.shape,
                    z_mask=analysis.masked_dataset.z_mask,
                )
                stats = _visibility_fit_stats(analysis, model_data)
                lens_peak = _peak_yx(lensed.sum(axis=0) * z_step, image_grid)
                peak_dist = float(
                    np.hypot(
                        lens_peak[0] - data_peak[0],
                        lens_peak[1] - data_peak[1],
                    )
                )
                rows.append(
                    {
                        "chi2_per_n": stats["chi_squared_per_datum"],
                        "peak_dist": peak_dist,
                        "phase": phase_name,
                        "phase_cent": phase_cent,
                        "src_flip": src_flip,
                        "post_flip": post_flip,
                        "lens_peak": lens_peak,
                    }
                )

    by_chi2 = sorted(rows, key=lambda r: r["chi2_per_n"])
    by_peak = sorted(rows, key=lambda r: r["peak_dist"])

    print(f"\n=== Top {top_n} by chi-squared / N ===\n")
    for row in by_chi2[:top_n]:
        print(
            f"  chi2/N={row['chi2_per_n']:.6f}  "
            f"peak_dist={row['peak_dist']:.3f}\"  "
            f"phase={row['phase']!r}  src={row['src_flip']}  "
            f"post={row['post_flip']}  lens_peak={row['lens_peak']}"
        )

    print(f"\n=== Top {min(8, top_n)} by peak distance ===\n")
    for row in by_peak[: min(8, top_n)]:
        print(
            f"  peak_dist={row['peak_dist']:.3f}\"  "
            f"chi2/N={row['chi2_per_n']:.6f}  "
            f"phase={row['phase']!r}  src={row['src_flip']}  "
            f"post={row['post_flip']}  lens_peak={row['lens_peak']}"
        )

    best = by_chi2[0]
    print("\n=== Recommendation ===\n")
    print(
        f"  Best UV: phase={best['phase']!r}, src_flip={best['src_flip']}, "
        f"post_flip={best['post_flip']} (chi2/N={best['chi2_per_n']:.6f})"
    )
    best_peak = by_peak[0]
    print(
        f"  Best peak: phase={best_peak['phase']!r}, src_flip={best_peak['src_flip']}, "
        f"post_flip={best_peak['post_flip']} (dist={best_peak['peak_dist']:.3f}\")"
    )
    print(
        "\n  Current production uses phaseCent [centre_0, centre_1] = (x, y), "
        "KinMS cube axes (v, y, x), and flip_kinms_y_before_lensing=True."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--run-phase1",
        action="store_true",
        help="Run phase-1 reconstruction (required for parametric_flux_from_phase1 / pixelized).",
    )
    parser.add_argument(
        "--lens-centre",
        choices=("fixed", "phase1"),
        default="fixed",
        help="Lens centre: JSON mass model (fixed) or phase-1 fit (phase1).",
    )
    args = parser.parse_args()
    run_trials(
        args.settings,
        top_n=args.top,
        run_phase1=args.run_phase1,
        lens_centre_mode=args.lens_centre,
    )


if __name__ == "__main__":
    main()
