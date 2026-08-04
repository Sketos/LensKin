#!/usr/bin/env python3
"""
Trial KinMS source-grid resolution and phase-1 regularization on a lensed mock.

For each ``(source_n_pixels, regularization [, image_mesh_shape])`` combo:

1. Run a fixed-lens phase-1 pixelization on moment-0 data (workspace-style fit).
2. Interpolate the reconstructed SB onto the KinMS source grid.
3. Forward-model **KinMSPixelized** at truth kinematics and score vs the mock
   (χ²/N, dirty-cube correlation, line-spectrum L1), with a frozen-cube ceiling
   and a frozen-SB baseline per source-grid size.

This extends the pixelized-truth diagnostic framework to quantify how phase-1
choices set the residual floor.

Examples::

  # Small smoke grid on the wide-velocity lensed mock
  python scripts/trial_source_grid_regularization.py \\
      --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \\
      --source-n-pixels 128,256 \\
      --reg 1e4,1e5

  # Broader scan including phase-1 mesh density
  python scripts/trial_source_grid_regularization.py \\
      --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \\
      --source-n-pixels 128,256,512 \\
      --reg 1e3,1e4,1e5,1e6 \\
      --mesh-shapes 20x20,30x30

  # Use Delaunay phase-1 mesh (heavier; settings default for production fits)
  python scripts/trial_source_grid_regularization.py \\
      --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \\
      --mesh delaunay --reg 1e5 --source-n-pixels 256

  # Frozen-cube SB only (no phase-1); isolates KinMS source-grid sampling
  python scripts/trial_source_grid_regularization.py \\
      --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \\
      --mode truth_sb \\
      --source-n-pixels 128,256,512
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
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

from scripts.generate_lensed_mock_pixelized_and_diagnose import (
    _pixelized_analysis_from_loaded,
    truth_sb_map_from_source_cube,
)
from scripts.test_truth_model import _visibility_fit_stats
from src.pipelines.phase1_likelihood import phase1_likelihood_breakdown
from src.pipelines.phase1_test import (
    adapt_images_for_pixelization,
    fixed_lens_galaxy_from_settings,
    interferometer_dataset_from_settings,
    pixelization_from_settings,
)
from src.pipelines.pixelized_plots import save_fit_triplet, save_image
from src.pipelines.reconstruction import (
    reconstruction_mask_from_settings,
    source_sb_from_fit,
    use_positive_only_solver_from_settings,
)
from src.pipelines.runner import load_cube_data, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils, plot_utils, spectral_utils


DEFAULT_SETTINGS = "settings/runners/kinms_mock_lensed_pixelized_widevel.json"


def _parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_mesh_shapes(text):
    """Parse ``20x20,30x30`` or ``20,20;30,30`` into list of (ny, nx)."""
    if text is None or not str(text).strip():
        return [None]
    shapes = []
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk.lower():
            a, b = chunk.lower().split("x", 1)
            shapes.append((int(a), int(b)))
        else:
            # allow "20 30" pairs by accumulating — prefer NxN tokens only
            raise ValueError(
                f"Mesh shape {chunk!r} must look like 30x30 (got comma-separated "
                "NxN tokens, e.g. '20x20,30x30')."
            )
    return shapes or [None]


def _settings_with_source_grid(
    settings, n_pixels, mesh_shape=None, reg_value=None, mesh_type=None
):
    settings = copy.deepcopy(settings)
    settings.setdefault("source_grid", {})
    settings["source_grid"]["n_pixels"] = int(n_pixels)
    rec = settings.setdefault("reconstruction", {})
    if mesh_type is not None:
        rec["mesh_type"] = mesh_type
        # constant_split is Delaunay-only; fall back to constant for rectangular.
        if mesh_type != "delaunay":
            reg = rec.setdefault("regularization", {})
            if reg.get("type", "constant") == "constant_split":
                reg["type"] = "constant"
    if mesh_shape is not None:
        rec["image_mesh_shape"] = [int(mesh_shape[0]), int(mesh_shape[1])]
        # Keep Delaunay edge density roughly matched to overlay mesh.
        rec["delaunay_edge_pixels"] = int(max(mesh_shape))
    if reg_value is not None:
        reg = rec.setdefault("regularization", {})
        reg["prior_type"] = "fixed"
        reg["value"] = float(reg_value)
    return settings


def _label(n_pixels, reg, mesh_shape):
    mesh = "default" if mesh_shape is None else f"{mesh_shape[0]}x{mesh_shape[1]}"
    reg_s = "none" if reg is None else f"{reg:.3g}"
    return f"src{n_pixels}_reg{reg_s}_mesh{mesh}"


def run_phase1_fit(settings, *, regularization_coefficient, output_dir=None):
    """Fixed-lens phase-1 fit; returns ``(fit, phase1_metrics)``."""
    import autolens as al

    mask_2d = reconstruction_mask_from_settings(settings)
    dataset = interferometer_dataset_from_settings(
        settings=settings,
        dataset_kind="moment0",
        transformer="auto",
    )
    use_jax = bool(settings.get("reconstruction", {}).get("use_jax", True))
    mesh_type = settings["reconstruction"].get("mesh_type", "delaunay")
    if mesh_type == "delaunay":
        use_jax = False
    if use_jax:
        dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=False)

    lens = fixed_lens_galaxy_from_settings(settings)
    pixelization, image_plane_mesh_grid = pixelization_from_settings(
        settings=settings,
        mask_2d=mask_2d,
        regularization_coefficient=regularization_coefficient,
        mesh_type=mesh_type,
    )
    source = al.Galaxy(
        redshift=settings["redshift_source"],
        pixelization=pixelization,
    )
    tracer = al.Tracer(galaxies=[lens, source])
    adapt_images = adapt_images_for_pixelization(
        source,
        image_plane_mesh_grid,
        dataset=dataset,
        settings=settings,
    )
    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_positive_only_solver=use_positive_only_solver_from_settings(settings)
        ),
    )
    breakdown = phase1_likelihood_breakdown(fit)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = autolens_utils.array2d_to_numpy(fit.dirty_image)
        model = autolens_utils.array2d_to_numpy(fit.dirty_model_image)
        _, _, width = autolens_utils.source_grid_from_settings(settings)
        extent = autolens_utils.image_extent_arcsec(width)
        save_fit_triplet(
            data,
            model,
            data - model,
            output_dir / "phase1_mom0_triplet.png",
            titles=("Dirty data", "Dirty model", "Residuals"),
            extent=extent,
            scale_mode="residual",
        )
        with open(output_dir / "phase1_summary.json", "w", encoding="utf-8") as handle:
            serializable = {}
            for key, value in breakdown.items():
                if value is None:
                    serializable[key] = None
                else:
                    try:
                        serializable[key] = float(value)
                    except (TypeError, ValueError):
                        serializable[key] = value
            json.dump(serializable, handle, indent=2)
    return fit, breakdown


def evaluate_pixelized_sb(
    settings,
    sb_map,
    frequencies,
    uv,
    vis,
    sigma,
    frozen_cube,
    *,
    plots_dir=None,
    tag="pixelized",
):
    """Score KinMSPixelized(truth kinematics, fixed SB) against mock visibilities."""
    settings = copy.deepcopy(settings)
    settings["model_name"] = "KinMSPixelized"
    z_step = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    analysis = _pixelized_analysis_from_loaded(
        settings, frequencies, uv, vis, sigma, sb_map
    )
    instance = truth_instance_from_settings(settings)
    model_vis = analysis.model_data_from_instance(instance=instance)
    stats = _visibility_fit_stats(analysis, model_vis)

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    dirty_model = autolens_utils.dirty_cube_from(
        visibilities=model_vis,
        transformers=analysis.transformers,
    )
    corr = float(np.corrcoef(dirty_data.ravel(), dirty_model.ravel())[0, 1])

    bulge = getattr(instance.galaxies.source, "bulge", instance.galaxies.source)
    pix_cube = bulge.profile_cube_from_masked_dataset(analysis.masked_dataset)
    # Resample frozen cube spatially if source grids differ.
    frozen = np.asarray(frozen_cube, dtype=float)
    if frozen.shape[1:] != pix_cube.shape[1:]:
        frozen_r = np.stack(
            [
                analysis_utils.resample_image_to_shape(ch, pix_cube.shape[1:])
                for ch in frozen
            ],
            axis=0,
        )
    else:
        frozen_r = frozen
    ft = frozen_r.sum(axis=(1, 2))
    pt = np.asarray(pix_cube, dtype=float).sum(axis=(1, 2))
    peak = float(ft.max()) if ft.size else 0.0
    line = ft > 0.01 * peak if peak > 0 else np.ones_like(ft, dtype=bool)
    if line.any() and pt[line].sum() > 0 and ft[line].sum() > 0:
        l1_line = float(
            np.sum(np.abs(pt[line] / pt[line].sum() - ft[line] / ft[line].sum()))
        )
    else:
        l1_line = float("nan")

    out = {
        "chi2_per_datum": float(stats["chi_squared_per_datum"]),
        "log_likelihood": float(stats["log_likelihood"]),
        "cube_corr": corr,
        "spectrum_l1_line": l1_line,
        "sb_sum": float(np.asarray(sb_map, dtype=float).sum()),
    }

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        extent = autolens_utils.image_extent_arcsec(settings["real_space_width"])
        mom0_d = dirty_data.sum(axis=0) * z_step
        mom0_m = dirty_model.sum(axis=0) * z_step
        dirty_sigma = autolens_utils.dirty_noise_cube_from(
            noise_map=sigma,
            transformers=analysis.transformers,
            n_realizations=8,
            seed=0,
        )
        mom0_sigma = autolens_utils.dirty_mom0_noise_from_channel_noise(
            dirty_sigma, z_step
        )
        save_fit_triplet(
            mom0_d,
            mom0_m,
            mom0_d - mom0_m,
            plots_dir / f"dirty_mom0_{tag}.png",
            titles=("Data dirty mom0", tag, "Data − model"),
            extent=extent,
            scale_mode="residual",
        )
        save_fit_triplet(
            mom0_d,
            mom0_m,
            mom0_d - mom0_m,
            plots_dir / f"dirty_mom0_{tag}_over_sigma.png",
            titles=("Data dirty mom0", tag, "Data − model"),
            extent=extent,
            scale_mode="sigma",
            residual_sigma=mom0_sigma,
        )
        save_image(
            sb_map,
            plots_dir / f"sb_map_{tag}.png",
            title=f"SB map ({tag})",
            extent=autolens_utils.image_extent_arcsec(
                settings.get("source_grid", {}).get(
                    "real_space_width", settings["real_space_width"]
                )
            ),
        )
        resid = dirty_data - dirty_model
        safe_sigma = np.where(
            np.isfinite(dirty_sigma) & (dirty_sigma > 0.0), dirty_sigma, np.nan
        )
        resid_over_sigma = resid / safe_sigma
        finite = resid_over_sigma[np.isfinite(resid_over_sigma)]
        rmin = float(np.min(finite)) if finite.size else float("nan")
        rmax = float(np.max(finite)) if finite.size else float("nan")
        peak = (
            float(np.nanmax(np.abs(resid_over_sigma)))
            if resid_over_sigma.size
            else 0.0
        )
        fig, _ = plot_utils.plot_cube(
            cube=resid_over_sigma,
            ncols=8,
            vmin=-5.0,
            vmax=5.0,
            cmap="RdBu_r",
            colorbar=True,
            colorbar_label=r"residual / $\sigma_{\mathrm{dirty}}$",
        )
        fig.suptitle(
            f"Channel residuals / σ_dirty ({tag}; colour ±5σ; "
            f"min={rmin:.2f}σ, max={rmax:.2f}σ, peak|r|={peak:.2f}σ)",
            y=1.02,
        )
        fig.savefig(
            plots_dir / f"channel_residuals_{tag}_over_sigma.png",
            bbox_inches="tight",
            dpi=120,
        )
        plt.close(fig)

    return out


def _frozen_ceiling(settings, frequencies, uv, vis, sigma, frozen_cube):
    from scripts.generate_lensed_mock_and_diagnose import _analysis_from_loaded

    analysis = _analysis_from_loaded(settings, frequencies, uv, vis, sigma)
    flip_y = bool(getattr(analysis, "_flip_kinms_y", True))
    lensed = analysis_utils.lensed_cube_from_tracer(
        cube=frozen_cube,
        tracer=analysis.tracer,
        grid=analysis.masked_dataset.grid_3d.grid_2d,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=analysis.masked_dataset.instance.grid_3d.grid_2d,
        output_shape=analysis.masked_dataset.grid_3d.shape_2d,
        flip_kinms_y=flip_y,
    )
    pb = getattr(analysis, "_primary_beam", None)
    frozen_vis = autolens_utils.visibilities_from_transformers_and_cube(
        cube=lensed,
        transformers=analysis.transformers,
        shape=analysis.masked_dataset.data.shape,
        z_mask=analysis.masked_dataset.z_mask,
        primary_beam=pb,
    )
    return _visibility_fit_stats(analysis, frozen_vis)


def _resample_sb_to_grid(sb_map, settings, n_channels):
    grid = autolens_utils.kinms_source_grid_3d(settings, n_channels=n_channels)
    target = grid.shape_2d
    sb = np.asarray(sb_map, dtype=float)
    if sb.shape == target:
        return sb, grid
    return analysis_utils.resample_image_to_shape(sb, target), grid


def run_trials(
    settings,
    *,
    source_n_pixels_list,
    reg_values,
    mesh_shapes,
    mode="phase1",
    mesh_type=None,
    plots=True,
    output_dir=None,
):
    settings = copy.deepcopy(settings)
    if mesh_type is not None:
        settings.setdefault("reconstruction", {})["mesh_type"] = mesh_type
        if mesh_type != "delaunay":
            reg = settings["reconstruction"].setdefault("regularization", {})
            if reg.get("type", "constant") == "constant_split":
                reg["type"] = "constant"
                print(
                    "  note: switched regularization.type constant_split → constant "
                    "(required for rectangular mesh)"
                )
    out = Path(
        output_dir
        if output_dir is not None
        else Path(settings["output_path"]) / "grid_reg_trials"
    )
    out.mkdir(parents=True, exist_ok=True)

    frequencies, uv, vis, sigma = load_cube_data(settings)
    n_chan = len(np.asarray(frequencies).reshape(-1))
    frozen_path = Path(settings["data_directory"]) / "frozen_source_cube.npy"
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"Need frozen source cube at {frozen_path} "
            "(generate the mock first, e.g. generate_lensed_mock_pixelized_and_diagnose.py)."
        )
    frozen_cube = np.load(frozen_path)
    truth_sb_native = truth_sb_map_from_source_cube(frozen_cube)
    active_mesh = settings.get("reconstruction", {}).get("mesh_type", "delaunay")

    print("\n=== Source-grid / regularization trials ===\n")
    print(f"  data = {settings['data_directory']}")
    print(f"  mode = {mode}")
    print(f"  phase-1 mesh_type = {active_mesh}")
    print(f"  source_n_pixels = {source_n_pixels_list}")
    print(f"  regularization = {reg_values if mode != 'truth_sb' else 'n/a'}")
    print(f"  mesh_shapes = {mesh_shapes}")
    print(f"  plots -> {out}")

    # Frozen ceiling once (uses settings image grid / UV; independent of source_grid).
    ceiling = _frozen_ceiling(settings, frequencies, uv, vis, sigma, frozen_cube)
    print(
        f"  frozen cube ceiling: chi2/N={ceiling['chi_squared_per_datum']:.6e}  "
        f"logL={ceiling['log_likelihood']:.2f}"
    )

    rows = []

    for n_pix in source_n_pixels_list:
        # Truth-SB baseline for this KinMS grid.
        s_truth = _settings_with_source_grid(settings, n_pix)
        sb_truth, _ = _resample_sb_to_grid(truth_sb_native, s_truth, n_chan)
        tag = _label(n_pix, None, None) + "_truthSB"
        trial_dir = out / tag if plots else None
        print(f"\n--- {tag} (frozen SB baseline) ---")
        metrics = evaluate_pixelized_sb(
            s_truth,
            sb_truth,
            frequencies,
            uv,
            vis,
            sigma,
            frozen_cube,
            plots_dir=trial_dir,
            tag="truthSB",
        )
        print(
            f"  chi2/N={metrics['chi2_per_datum']:.6e}  corr={metrics['cube_corr']:.4f}  "
            f"L1_line={metrics['spectrum_l1_line']:.4f}"
        )
        rows.append(
            {
                "mode": "truth_sb",
                "source_n_pixels": n_pix,
                "regularization": "",
                "mesh_shape": "",
                "phase1_chi2": "",
                "phase1_log_evidence": "",
                **metrics,
                "tag": tag,
            }
        )

        if mode == "truth_sb":
            continue

        for mesh_shape in mesh_shapes:
            for reg in reg_values:
                tag = _label(n_pix, reg, mesh_shape)
                trial_dir = out / tag if plots else None
                print(f"\n--- {tag} ---")
                s = _settings_with_source_grid(
                    settings,
                    n_pix,
                    mesh_shape=mesh_shape,
                    reg_value=reg,
                    mesh_type=mesh_type,
                )
                try:
                    fit, p1 = run_phase1_fit(
                        s,
                        regularization_coefficient=reg,
                        output_dir=trial_dir,
                    )
                    grid = autolens_utils.kinms_source_grid_3d(s, n_channels=n_chan)
                    sb_map = source_sb_from_fit(fit, grid.grid_2d)
                    if trial_dir is not None:
                        np.save(trial_dir / "phase1_sb_on_kinms_grid.npy", sb_map)

                    metrics = evaluate_pixelized_sb(
                        s,
                        sb_map,
                        frequencies,
                        uv,
                        vis,
                        sigma,
                        frozen_cube,
                        plots_dir=trial_dir,
                        tag="phase1SB",
                    )
                    print(
                        f"  phase1 chi2={p1['chi_squared']:.2f}  "
                        f"log_evidence={p1['figure_of_merit_log_evidence']:.2f}"
                    )
                    print(
                        f"  phase2 chi2/N={metrics['chi2_per_datum']:.6e}  "
                        f"corr={metrics['cube_corr']:.4f}  "
                        f"L1_line={metrics['spectrum_l1_line']:.4f}"
                    )
                    rows.append(
                        {
                            "mode": "phase1",
                            "source_n_pixels": n_pix,
                            "regularization": reg,
                            "mesh_shape": (
                                ""
                                if mesh_shape is None
                                else f"{mesh_shape[0]}x{mesh_shape[1]}"
                            ),
                            "phase1_chi2": p1["chi_squared"],
                            "phase1_log_evidence": p1["figure_of_merit_log_evidence"],
                            **metrics,
                            "tag": tag,
                        }
                    )
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    rows.append(
                        {
                            "mode": "phase1",
                            "source_n_pixels": n_pix,
                            "regularization": reg,
                            "mesh_shape": (
                                ""
                                if mesh_shape is None
                                else f"{mesh_shape[0]}x{mesh_shape[1]}"
                            ),
                            "phase1_chi2": "FAILED",
                            "phase1_log_evidence": "FAILED",
                            "chi2_per_datum": float("nan"),
                            "log_likelihood": float("nan"),
                            "cube_corr": float("nan"),
                            "spectrum_l1_line": float("nan"),
                            "sb_sum": float("nan"),
                            "tag": tag,
                            "error": str(exc),
                        }
                    )
                    if trial_dir is not None:
                        with open(trial_dir / "error.txt", "w", encoding="utf-8") as handle:
                            handle.write(str(exc))

    csv_path = out / "trial_summary.csv"
    fieldnames = [
        "mode",
        "source_n_pixels",
        "regularization",
        "mesh_shape",
        "phase1_chi2",
        "phase1_log_evidence",
        "chi2_per_datum",
        "log_likelihood",
        "cube_corr",
        "spectrum_l1_line",
        "sb_sum",
        "tag",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "frozen_ceiling": {
            "chi_squared_per_datum": float(ceiling["chi_squared_per_datum"]),
            "log_likelihood": float(ceiling["log_likelihood"]),
        },
        "n_trials": len(rows),
        "csv": str(csv_path),
    }
    with open(out / "trial_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _plot_summary_heatmaps(rows, out, ceiling_chi2=float(ceiling["chi_squared_per_datum"]))
    print(f"\n  Wrote {csv_path}")
    print(f"  Done. Inspect {out}")
    return rows, out


def _plot_summary_heatmaps(rows, out_dir, *, ceiling_chi2):
    """Heatmaps of phase-2 χ²/N vs (source_n_pixels, regularization) when possible."""
    phase1_rows = [
        r
        for r in rows
        if r.get("mode") == "phase1"
        and r.get("regularization") not in ("", None)
        and np.isfinite(float(r.get("chi2_per_datum", np.nan)))
    ]
    if not phase1_rows:
        return

    # Prefer a single mesh_shape slice (default / first).
    meshes = sorted({r.get("mesh_shape") or "default" for r in phase1_rows})
    for mesh in meshes:
        subset = [
            r
            for r in phase1_rows
            if (r.get("mesh_shape") or "default") == mesh
        ]
        srcs = sorted({int(r["source_n_pixels"]) for r in subset})
        regs = sorted({float(r["regularization"]) for r in subset})
        if not srcs or not regs:
            continue
        mat = np.full((len(srcs), len(regs)), np.nan)
        for r in subset:
            i = srcs.index(int(r["source_n_pixels"]))
            j = regs.index(float(r["regularization"]))
            mat[i, j] = float(r["chi2_per_datum"])

        fig, ax = plt.subplots(figsize=(1.2 * len(regs) + 2, 1.0 * len(srcs) + 2))
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([f"{r:.3g}" for r in regs], rotation=45, ha="right")
        ax.set_yticks(range(len(srcs)))
        ax.set_yticklabels([str(s) for s in srcs])
        ax.set_xlabel("regularization coefficient")
        ax.set_ylabel("source_grid n_pixels")
        ax.set_title(
            f"Phase-2 χ²/N (mesh={mesh}; frozen ceiling={ceiling_chi2:.3g})"
        )
        for i in range(len(srcs)):
            for j in range(len(regs)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe = mesh.replace("/", "_")
        fig.savefig(out_dir / f"heatmap_chi2_mesh_{safe}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--mode",
        choices=("phase1", "truth_sb", "both"),
        default="phase1",
        help="phase1: reconstruct SB then score; truth_sb: frozen SB only; both: run phase1",
    )
    parser.add_argument(
        "--source-n-pixels",
        default="128,256",
        help="Comma-separated KinMS source_grid n_pixels values",
    )
    parser.add_argument(
        "--reg",
        default="1e4,1e5",
        help="Comma-separated phase-1 regularization coefficients",
    )
    parser.add_argument(
        "--mesh-shapes",
        default="",
        help="Optional comma-separated phase-1 overlay meshes, e.g. 20x20,30x30 "
        "(empty = keep settings image_mesh_shape)",
    )
    parser.add_argument(
        "--mesh",
        choices=("rectangular", "delaunay", "settings"),
        default="rectangular",
        help="Phase-1 mesh (default rectangular — much lighter than delaunay for scans)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-trial diagnostic PNGs (still writes summary CSV/heatmap)",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    mode = "phase1" if args.mode == "both" else args.mode
    mesh_type = None
    if args.mesh == "rectangular":
        mesh_type = "rectangular_adapt_density"
    elif args.mesh == "delaunay":
        mesh_type = "delaunay"
    run_trials(
        settings,
        source_n_pixels_list=_parse_int_list(args.source_n_pixels),
        reg_values=_parse_float_list(args.reg),
        mesh_shapes=_parse_mesh_shapes(args.mesh_shapes),
        mode=mode,
        mesh_type=mesh_type,
        plots=not args.no_plots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
