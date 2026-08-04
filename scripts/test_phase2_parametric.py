#!/usr/bin/env python3
"""
Phase-2 parametric test: truth kinematics + SB, free source centre only.

Same workflow as ``test_phase2_pixelization.py``, but phase 2 uses the
**parametric** KinMS exponential-disk model (``effective_radius`` stays
parametric) while taking the **total flux / intensity from phase 1**.

  1. Phase 1 — pixelized SB reconstruction with **free lens centre** (identical
     to the pixelized test; only the fitted lens centre is passed forward).
  2. Phase 2 — parametric ``kinMS`` with **all truth parameters fixed** except
     ``centre_0`` / ``centre_1``, and ``intensity`` fixed to the phase-1 total
     flux (mode-2 / ``parametric_flux_from_phase1`` behavior).

Examples:
  python scripts/test_phase2_parametric.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json

  python scripts/test_phase2_parametric.py \\
    --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \\
    --run-nautilus
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

from src.model import profiles
from src.pipelines.phase2_test import (
    build_phase2_analysis_parametric,
    optimize_source_centre,
    phase2_output_dir,
    print_centre_optimization_report,
    print_truth_parameter_summary,
    run_phase1_for_lens_centre,
    run_phase2_nautilus,
    save_phase2_plots,
)
from src.pipelines.pixelized_plots import plot_phase2_fit
from src.pipelines.runner import load_settings


def run_test(
    settings,
    *,
    phase1_only=False,
    run_nautilus=False,
    save_plots=True,
):
    phase2_cfg = settings.get("phase2", {})
    free_keys = tuple(phase2_cfg.get("free_parameters", ("centre_0", "centre_1")))

    print("\n=== Phase-2 parametric test setup ===\n")
    truth_params = print_truth_parameter_summary(settings, free_keys)

    out_root = phase2_output_dir(settings)
    phase1_bundle = run_phase1_for_lens_centre(
        settings, output_dir=out_root if save_plots else None
    )

    if phase1_only:
        print(f"\n=== Phase 1 complete (plots under {out_root / 'phase1'}) ===\n")
        return {"phase1": phase1_bundle}

    print(
        "\n=== Phase 2: parametric KinMS "
        "(phase-1 total flux + truth shape/kinematics, free centre) ===\n"
    )
    print(
        f"  Lens centre (from phase 1) = "
        f"({phase1_bundle['lens_centre'][0]:.4f}, {phase1_bundle['lens_centre'][1]:.4f})"
    )
    print(
        f"  Parametric SB: phase-1 intensity = "
        f"{phase1_bundle['phase1_total_flux']!r} Jy km/s, "
        f"effective_radius = {truth_params['effective_radius']!r} arcsec"
    )

    analysis_instance = build_phase2_analysis_parametric(settings, phase1_bundle)
    centre_report = optimize_source_centre(
        settings, analysis_instance, truth_params
    )
    print_centre_optimization_report(centre_report)

    tc = centre_report["truth_centre"]
    bc = centre_report["best_centre"]
    centre_dist = float(np.hypot(bc[0] - tc[0], bc[1] - tc[1]))
    best_c2n = centre_report["best_stats"]["chi_squared_per_datum"]
    truth_c2n = centre_report["truth_stats"]["chi_squared_per_datum"]

    if centre_dist < 0.05 and best_c2n < 1.2:
        print("\n  PASS — truth centre near optimum with parametric SB + phase-1 lens.")
    elif best_c2n < truth_c2n - 1e-4:
        print(
            "\n  CENTRE OFFSET — a different source centre improves the UV fit; "
            "compare with the pixelized phase-2 test to isolate SB vs kinematics."
        )
    else:
        print(
            "\n  REVIEW — even with phase-1 total flux, the parametric model may "
            "still have lens-centre or SB-shape mismatches."
        )

    result = {
        "phase1": phase1_bundle,
        "centre_optimization": centre_report,
    }

    if run_nautilus:
        print("\n=== Phase 2: Nautilus (truth params, free centre only) ===\n")
        nautilus_result = run_phase2_nautilus(
            settings, analysis_instance, profiles.kinMS
        )
        result["nautilus"] = nautilus_result
        centre = nautilus_result.max_log_likelihood_instance.galaxies.source.centre
        print(f"  Nautilus MAP centre = ({centre[0]:.4f}, {centre[1]:.4f})")
        if save_plots:
            plot_phase2_fit(
                analysis_instance=analysis_instance,
                result=nautilus_result,
                output_dir=out_root / "phase2_nautilus",
                settings=settings,
            )

    if save_plots:
        save_phase2_plots(
            analysis_instance,
            centre_report["best_model_data"],
            out_root / "phase2_optimized_centre",
        )
        print(f"\n=== Plots saved under {out_root} ===\n")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--phase1-only", action="store_true")
    parser.add_argument("--run-nautilus", action="store_true")
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
