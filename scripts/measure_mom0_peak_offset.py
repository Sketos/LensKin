#!/usr/bin/env python3
"""
Measure dirty mom0 peak offset between data and the truth model.

Reports the Autolens (y, x) peak locations, the model−data offset in arcsec,
and how many image-plane / source-plane pixels that offset is. Optionally
nudges lens or source centre by ±1 pixel to see which axis shrinks residuals.

Examples:
  python scripts/measure_mom0_peak_offset.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --run-phase1 --lens-centre phase1

  python scripts/measure_mom0_peak_offset.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --run-phase1 --lens-centre fixed --nudge
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

import numpy as np

from scripts.test_truth_model import (
    _build_pipeline,
    _visibility_fit_stats,
)
from src.analysis import analysis as analysis_mod
from src.pipelines.pixelized_plots import save_fit_triplet
from src.pipelines.runner import build_tracer, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils


def _peak_yx(mom0, grid_2d):
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    i, j = np.unravel_index(np.nanargmax(mom0), mom0.shape)
    return float(coords[i, j, 0]), float(coords[i, j, 1]), int(i), int(j)


def _pixel_scales_yx(grid_2d):
    shape = grid_2d.shape_native
    coords = np.asarray(grid_2d).reshape(shape + (2,))
    dy = abs(float(coords[1, 0, 0] - coords[0, 0, 0])) if shape[0] > 1 else float("nan")
    dx = abs(float(coords[0, 1, 1] - coords[0, 0, 1])) if shape[1] > 1 else float("nan")
    return dy, dx


def _analysis_with_tracer(pipeline, settings, tracer):
    return analysis_mod.Analysis(
        masked_dataset=pipeline["analysis"].masked_dataset,
        transformers=pipeline["analysis"].transformers,
        tracer=tracer,
        settings=settings,
    )


def _mom0_from_visibilities(visibilities, transformers, z_step_kms):
    dirty = autolens_utils.dirty_cube_from(
        visibilities=visibilities,
        transformers=transformers,
    )
    return np.asarray(dirty).sum(axis=0) * float(z_step_kms)


def _evaluate(analysis, instance, image_grid, source_grid, z_step, settings):
    flip_kinms_y = bool(settings.get("flip_kinms_y_before_lensing", True))
    source_cube = analysis.model_cube_from_instance(instance=instance)
    lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=source_cube,
        tracer=analysis.tracer,
        grid=image_grid,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=source_grid,
        output_shape=image_grid.shape_native,
        flip_kinms_y=flip_kinms_y,
    )
    model_data = analysis.model_data_from_instance(instance=instance)
    stats = _visibility_fit_stats(analysis, model_data)
    model_dirty_mom0 = _mom0_from_visibilities(
        model_data, analysis.transformers, z_step
    )
    source_mom0 = np.asarray(source_cube).sum(axis=0) * float(z_step)
    lensed_mom0 = np.asarray(lensed_cube).sum(axis=0) * float(z_step)
    return {
        "stats": stats,
        "source_mom0": source_mom0,
        "lensed_mom0": lensed_mom0,
        "model_dirty_mom0": model_dirty_mom0,
        "model_data": model_data,
    }


def _report_peaks(label, data_mom0, model_mom0, image_grid, source_pix, img_pix):
    dy_img, dx_img = _pixel_scales_yx(image_grid)
    data_y, data_x, di, dj = _peak_yx(data_mom0, image_grid)
    model_y, model_x, mi, mj = _peak_yx(model_mom0, image_grid)
    dy = model_y - data_y
    dx = model_x - data_x
    dist = float(np.hypot(dy, dx))

    print(f"\n=== {label} ===")
    print(f"  data  dirty mom0 peak (y, x) = ({data_y:+.6f}, {data_x:+.6f})  pixel=[{di}, {dj}]")
    print(f"  model dirty mom0 peak (y, x) = ({model_y:+.6f}, {model_x:+.6f})  pixel=[{mi}, {mj}]")
    print(f"  Δ peak (model − data)        = ({dy:+.6f}, {dx:+.6f}) arcsec")
    print(f"  |Δ|                          = {dist:.6f} arcsec")
    print(
        f"  |Δ| / image pixel             = {dist / img_pix:.3f}   "
        f"(image Δy={dy_img:.5f}\", Δx={dx_img:.5f}\")"
    )
    print(f"  |Δ| / source pixel            = {dist / source_pix:.3f}   (source pixel={source_pix:.5f}\")")
    print(f"  peak pixel index Δ (row,col) = ({mi - di:+d}, {mj - dj:+d})")
    return {
        "data_peak": (data_y, data_x),
        "model_peak": (model_y, model_x),
        "delta_yx": (dy, dx),
        "dist": dist,
    }


def _nudge_scan(pipeline, settings, image_grid, source_grid, z_step, data_mom0, img_pix, source_pix):
    """Evaluate UV chi² and peak |Δ| for ±1 image / source pixel nudges."""
    base_lens = pipeline["lens_centre"]
    base_instance = truth_instance_from_settings(settings)
    base_centre = base_instance.galaxies.source.centre

    trials = [("baseline", "none", 0.0, 0.0)]
    for axis, sign in (("y", +1), ("y", -1), ("x", +1), ("x", -1)):
        trials.append((f"lens ±1 img pix ({axis}{sign:+d})", "lens", axis, sign * img_pix))
        trials.append(
            (f"source ±1 src pix ({axis}{sign:+d})", "source", axis, sign * source_pix)
        )

    print("\n=== Nudge scan (hold other parameters at truth) ===\n")
    print(
        f"{'label':40s}  {'chi2/N':>10s}  {'|Δpeak|':>10s}  "
        f"{'Δy':>10s}  {'Δx':>10s}"
    )
    rows = []
    for label, kind, axis, delta in trials:
        instance = truth_instance_from_settings(settings)
        if kind == "none":
            analysis = pipeline["analysis"]
            if getattr(analysis, "settings", None) is None:
                analysis = _analysis_with_tracer(
                    pipeline, settings, pipeline["analysis"].tracer
                )
        elif kind == "lens":
            c0, c1 = base_lens
            if axis == "y":
                c0 = float(c0) + float(delta)
            else:
                c1 = float(c1) + float(delta)
            tracer = build_tracer(settings, centre=(c0, c1))
            analysis = _analysis_with_tracer(pipeline, settings, tracer)
        else:
            c0, c1 = base_centre
            # source centre tuple is (x, y) = (centre_0, centre_1)
            if axis == "y":
                c1 = float(c1) + float(delta)
            else:
                c0 = float(c0) + float(delta)
            instance.galaxies.source.centre = (c0, c1)
            analysis = pipeline["analysis"]
            if getattr(analysis, "settings", None) is None:
                analysis = _analysis_with_tracer(
                    pipeline, settings, pipeline["analysis"].tracer
                )

        result = _evaluate(
            analysis, instance, image_grid, source_grid, z_step, settings
        )
        peaks = _peak_yx(result["model_dirty_mom0"], image_grid)
        data_peak = _peak_yx(data_mom0, image_grid)
        dy = peaks[0] - data_peak[0]
        dx = peaks[1] - data_peak[1]
        dist = float(np.hypot(dy, dx))
        chi2n = result["stats"]["chi_squared_per_datum"]
        print(
            f"{label:40s}  {chi2n:10.6f}  {dist:10.6f}  "
            f"{dy:+10.6f}  {dx:+10.6f}"
        )
        rows.append((label, chi2n, dist, dy, dx))

    best = min(rows, key=lambda r: r[1])
    print(
        f"\n  Best UV chi²/N: {best[0]!r} "
        f"(chi2/N={best[1]:.6f}, |Δpeak|={best[2]:.6f}\")"
    )
    return rows


def run(
    settings_path,
    *,
    run_phase1=False,
    lens_centre_mode="fixed",
    nudge=False,
    plots_dir=None,
):
    settings = load_settings(settings_path)
    # Ensure Analysis sees flip / free-lens settings.
    settings = copy.deepcopy(settings)

    pipeline = _build_pipeline(
        settings,
        run_phase1=run_phase1,
        lens_centre_mode=lens_centre_mode,
    )
    # Re-wrap analysis with settings so flip_kinms_y_before_lensing is honoured.
    analysis = _analysis_with_tracer(
        pipeline, settings, pipeline["analysis"].tracer
    )
    pipeline = dict(pipeline)
    pipeline["analysis"] = analysis

    image_grid = pipeline["image_grid_3d"].grid_2d
    source_grid = pipeline["kinms_grid_3d"].grid_2d
    z_step = pipeline["z_step_kms"]
    instance = truth_instance_from_settings(settings)

    img_n, img_pix, _ = autolens_utils.image_plane_grid_from_settings(settings)
    src_n, src_pix, _ = autolens_utils.source_grid_from_settings(settings)
    # Prefer KinMS source grid pixel scale when phase-1 bbox is used.
    src_dy, src_dx = _pixel_scales_yx(source_grid)
    source_pix = float(np.nanmean([src_dy, src_dx]))
    if not np.isfinite(source_pix) or source_pix <= 0.0:
        source_pix = float(src_pix)

    data_mom0 = _mom0_from_visibilities(
        analysis.masked_dataset.data,
        analysis.transformers,
        z_step,
    )

    print("\n=== Mom0 peak offset test ===\n")
    print(f"  settings              = {settings_path}")
    print(f"  run_phase1            = {run_phase1}")
    print(f"  lens_centre_mode      = {lens_centre_mode}")
    print(f"  lens centre           = {pipeline['lens_centre']}")
    print(
        f"  flip_kinms_y_before_lensing = "
        f"{settings.get('flip_kinms_y_before_lensing', True)}"
    )
    print(f"  image grid            = {img_n}², pixel={img_pix:.6f}\"")
    print(
        f"  source / KinMS grid    = {source_grid.shape_native[0]}², "
        f"pixel≈{source_pix:.6f}\""
    )
    print(
        f"  source centre (x, y)   = {instance.galaxies.source.centre}"
    )

    result = _evaluate(
        analysis, instance, image_grid, source_grid, z_step, settings
    )
    stats = result["stats"]
    print("\n=== UV fit at truth ===")
    print(f"  chi-squared / N = {stats['chi_squared_per_datum']:.6f}")
    print(f"  log L           = {stats['log_likelihood']:.2f}")
    print(f"  residual RMS    = {stats['residual_rms']:.6g}")

    _report_peaks(
        "Dirty mom0 peaks (UV → dirty image)",
        data_mom0,
        result["model_dirty_mom0"],
        image_grid,
        source_pix,
        float(img_pix),
    )
    _report_peaks(
        "Lensed model cube mom0 vs dirty data",
        data_mom0,
        result["lensed_mom0"],
        image_grid,
        source_pix,
        float(img_pix),
    )

    src_y, src_x, _, _ = _peak_yx(result["source_mom0"], source_grid)
    print("\n=== Source-plane model mom0 ===")
    print(f"  source mom0 peak (y, x) = ({src_y:+.6f}, {src_x:+.6f})")

    out_dir = Path(
        plots_dir
        if plots_dir is not None
        else Path(settings["output_path"]) / "mom0_peak_offset"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, real_space_width = autolens_utils.image_plane_grid_from_settings(settings)
    extent = autolens_utils.image_extent_arcsec(real_space_width)
    residual = data_mom0 - result["model_dirty_mom0"]
    save_fit_triplet(
        data_mom0,
        result["model_dirty_mom0"],
        residual,
        out_dir / "dirty_mom0_data_model_residual.png",
        titles=("Data dirty mom0", "Model dirty mom0", "Data − model"),
        extent=extent,
    )
    print(f"\n  Wrote {out_dir / 'dirty_mom0_data_model_residual.png'}")

    if nudge:
        _nudge_scan(
            pipeline,
            settings,
            image_grid,
            source_grid,
            z_step,
            data_mom0,
            float(img_pix),
            source_pix,
        )

    print()
    return {
        "stats": stats,
        "data_mom0": data_mom0,
        "model_dirty_mom0": result["model_dirty_mom0"],
        "output_dir": out_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--settings", required=True)
    parser.add_argument(
        "--run-phase1",
        action="store_true",
        help="Run phase-1 (required for parametric_flux_from_phase1 / pixelized).",
    )
    parser.add_argument(
        "--lens-centre",
        choices=("fixed", "phase1"),
        default="fixed",
        help="Lens centre from JSON (fixed) or phase-1 fit (phase1).",
    )
    parser.add_argument(
        "--nudge",
        action="store_true",
        help="Scan ±1 image-pixel lens and ±1 source-pixel source centre nudges.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Output directory for mom0 triplet (default: <output_path>/mom0_peak_offset).",
    )
    args = parser.parse_args()
    run(
        args.settings,
        run_phase1=args.run_phase1,
        lens_centre_mode=args.lens_centre,
        nudge=args.nudge,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
