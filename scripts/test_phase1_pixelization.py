#!/usr/bin/env python
"""
Test phase-1 pixelized source reconstruction with a fixed lens model.

This script mirrors
``~/Work/autolens_workspace/scripts/interferometer/features/pixelization/fit.py``,
but loads LensKin exported FITS cubes and uses the ``lens_mass_model`` block from
a runner settings JSON (e.g. ``kinms_mock_pixelized.json`` or
``SPT0538_CO9-8_pixelized_test.json``).

Usage (from the LensKin repo root):

  python scripts/test_phase1_pixelization.py \\
      --settings settings/runners/kinms_mock_pixelized.json

  python scripts/test_phase1_pixelization.py \\
      --settings settings/runners/kinms_mock_pixelized.json \\
      --mesh delaunay

  python scripts/test_phase1_pixelization.py \\
      --settings settings/runners/kinms_mock_pixelized.json \\
      --mode both --mesh delaunay --dataset channel

Modes:
  fixed_lens  workspace-style ``FitInterferometer`` with fixed lens (default)
  pipeline    LensKin ``run_reconstruction`` LBFGS search
  both        run both and save plots under separate subfolders

Mesh types (``--mesh``):
  rectangular  rectangular adapt-density mesh (settings default)
  delaunay     Delaunay triangulation with Overlay image-mesh (workspace delaunay.py)
"""

import argparse
import copy
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

import autofit as af

from scripts.bootstrap import setup
from src.pipelines import reconstruction
from src.pipelines.phase1_test import run_workspace_style_fit
from src.pipelines.pixelized_plots import plot_phase1_fit
from src.pipelines.phase1_likelihood import (
    format_likelihood_breakdown,
    phase1_likelihood_breakdown,
)
from src.pipelines.runner import load_settings
from src.utils import autolens_utils

ROOT = setup(__file__)

_MESH_CHOICES = {
    "rectangular": "rectangular_adapt_density",
    "delaunay": "delaunay",
}


def _settings_with_mesh(settings, mesh):
    if mesh is None:
        return settings
    settings = copy.deepcopy(settings)
    settings.setdefault("reconstruction", {})
    settings["reconstruction"]["mesh_type"] = _MESH_CHOICES[mesh]
    return settings


def _mesh_output_name(mesh):
    return mesh or "rectangular"


def _regularization_overrides_from_args(args, reg_type):
    """Map CLI flags to regularization parameter overrides."""
    overrides = {}
    if args.inner_regularization is not None:
        overrides["inner_coefficient"] = args.inner_regularization
    if args.outer_regularization is not None:
        overrides["outer_coefficient"] = args.outer_regularization
    if args.signal_scale is not None:
        overrides["signal_scale"] = args.signal_scale
    if args.regularization is not None:
        if reg_type in reconstruction._ADAPT_REGULARIZATION_TYPES:
            overrides.setdefault("outer_coefficient", args.regularization)
            overrides.setdefault(
                "inner_coefficient", args.regularization * 0.0026
            )
        else:
            overrides["coefficient"] = args.regularization
    return overrides


def _regularization_summary(settings, overrides):
    rec_cfg = settings["reconstruction"]
    reg_cfg = rec_cfg.get("regularization", {})
    reg_type = reconstruction.regularization_type_from_settings(rec_cfg)
    if overrides:
        values = reconstruction.fixed_regularization_values_from_settings(
            reg_cfg, reg_type, overrides=overrides
        )
        summary = reconstruction.format_regularization_values(reg_type, values)
    else:
        summary = reconstruction.format_regularization_config(reg_cfg, reg_type)
    return reg_type, summary


def _run_pipeline_style(settings, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = copy.deepcopy(settings)
    settings["output_path"] = str(output_dir)
    settings.setdefault("reconstruction", {})
    settings["reconstruction"]["visualize"] = False
    settings["reconstruction"].setdefault("fix_lens", True)

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    result = reconstruction.run_reconstruction(settings)

    from src.pipelines.runner import load_cube_data

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

    plot_phase1_fit(
        result=result,
        sb_map=sb_map,
        output_dir=output_dir,
        settings=settings,
        sb_map_extent=autolens_utils.image_extent_from_bounding_box(
            reconstruction.source_plane_bounding_box_from_result(result)
        ),
    )

    instance = result.max_log_likelihood_instance
    mass_centre = instance.galaxies.lens.mass.centre
    mesh_type = settings["reconstruction"].get("mesh_type", "rectangular_adapt_density")
    reg = instance.galaxies.source.pixelization.regularization
    reg_type = reconstruction.regularization_type_from_settings(
        settings["reconstruction"]
    )
    if hasattr(reg, "coefficient"):
        reg_summary = f"coefficient={reg.coefficient}"
    else:
        reg_summary = (
            f"inner_coefficient={reg.inner_coefficient}, "
            f"outer_coefficient={reg.outer_coefficient}, "
            f"signal_scale={reg.signal_scale}"
        )
    breakdown = phase1_likelihood_breakdown(result.max_log_likelihood_fit)
    with open(output_dir / "fit_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("pipeline_mode = run_reconstruction\n")
        handle.write(f"mesh_type = {mesh_type}\n")
        handle.write(f"regularization_type = {reg_type}\n")
        handle.write(f"regularization = {reg_summary}\n")
        handle.write(f"fitted_lens_centre = ({mass_centre[0]}, {mass_centre[1]})\n")
        handle.write(f"log_evidence = {breakdown['figure_of_merit_log_evidence']}\n")
        handle.write(f"log_likelihood_data = {breakdown['log_likelihood_data']}\n")
        handle.write(
            f"log_likelihood_with_regularization = "
            f"{breakdown['log_likelihood_with_regularization']}\n"
        )

    return result, output_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test phase-1 pixelized reconstruction with a fixed lens model "
            "(autolens_workspace pixelization/fit.py style)."
        )
    )
    parser.add_argument(
        "--settings",
        default=str(ROOT / "settings" / "runners" / "kinms_mock_pixelized.json"),
        help="Runner settings JSON with lens_mass_model and reconstruction block",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed_lens", "pipeline", "both"),
        default="fixed_lens",
        help="fixed_lens: FitInterferometer; pipeline: run_reconstruction; both",
    )
    parser.add_argument(
        "--dataset",
        choices=("moment0", "channel"),
        default="moment0",
        help="moment0: velocity-averaged visibilities; channel: single channel",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Spectral channel index when --dataset=channel (default: middle channel)",
    )
    parser.add_argument(
        "--transformer",
        choices=("nufft", "dft"),
        default="nufft",
        help="UV transform: NUFFT (LensKin default) or DFT (workspace default)",
    )
    parser.add_argument(
        "--mesh",
        choices=sorted(_MESH_CHOICES),
        default=None,
        help=(
            "Pixelization mesh: rectangular (default from settings) or delaunay "
            "(Overlay image-mesh + Delaunay triangulation)"
        ),
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=None,
        help=(
            "For constant/constant_split: sets coefficient. For adapt/adapt_split: "
            "sets outer_coefficient (inner defaults to outer*0.0026 unless "
            "--inner-regularization is set). Omit to use settings JSON values."
        ),
    )
    parser.add_argument(
        "--inner-regularization",
        type=float,
        default=None,
        help="Adapt/adapt_split inner_coefficient (bright regions, less smoothing)",
    )
    parser.add_argument(
        "--outer-regularization",
        type=float,
        default=None,
        help="Adapt/adapt_split outer_coefficient (blank regions, more smoothing)",
    )
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=None,
        help="Adapt/adapt_split signal_scale (steepness of bright/blank transition)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "phase1_test"),
        help="Directory for diagnostic plots and summary files",
    )
    parser.add_argument(
        "--no-jax",
        action="store_true",
        help="Disable sparse-operator JAX acceleration",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable sparse-operator progress bar",
    )
    args = parser.parse_args()

    settings = _settings_with_mesh(load_settings(args.settings), args.mesh)
    output_root = Path(args.output_dir)
    mass_cfg = settings["lens_mass_model"]
    mesh_label = _mesh_output_name(args.mesh)
    mesh_type = settings["reconstruction"].get("mesh_type", "rectangular_adapt_density")
    reg_type = reconstruction.regularization_type_from_settings(
        settings["reconstruction"]
    )
    reg_overrides = _regularization_overrides_from_args(args, reg_type)
    _, reg_summary = _regularization_summary(settings, reg_overrides)

    print("Phase-1 pixelization test")
    print(f"  settings: {args.settings}")
    print(f"  mode: {args.mode}")
    print(f"  mesh: {mesh_type}")
    print(f"  regularization_type: {reg_type}")
    print(f"  regularization: {reg_summary}")
    print(f"  dataset: {args.dataset}")
    print(f"  transformer: {args.transformer}")
    print(f"  lens centre: ({mass_cfg['centre_0']}, {mass_cfg['centre_1']})")
    print(f"  einstein_radius: {mass_cfg['einstein_radius']}")
    print(f"  output: {output_root}")

    if args.mode in {"fixed_lens", "both"}:
        fit, out_dir = run_workspace_style_fit(
            settings=settings,
            output_dir=output_root / f"fixed_lens_{mesh_label}",
            dataset_kind=args.dataset,
            channel=args.channel,
            transformer=args.transformer,
            regularization_coefficient=args.regularization,
            mesh_type=mesh_type,
            regularization_overrides=reg_overrides,
            use_jax=not args.no_jax,
            show_progress=not args.no_progress,
        )
        print(f"Fixed-lens fit complete: {format_likelihood_breakdown(phase1_likelihood_breakdown(fit))}")
        print(f"  plots: {out_dir}")

    if args.mode in {"pipeline", "both"}:
        _, out_dir = _run_pipeline_style(
            settings=settings,
            output_dir=output_root / f"pipeline_{mesh_label}",
        )
        print("Pipeline reconstruction complete")
        print(f"  plots: {out_dir}")


if __name__ == "__main__":
    main()
