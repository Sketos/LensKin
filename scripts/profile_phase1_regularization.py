#!/usr/bin/env python
"""
Profile phase-1 figures of merit vs regularization coefficient.

**Important:** ``fit.log_likelihood`` is the data-only term and always favours
low regularization (overfitting). LBFGS maximizes ``fit.figure_of_merit``
(``log_evidence``), which includes the regularization penalty and Occam factors.

Examples::

  python scripts/profile_phase1_regularization.py \\
      --settings settings/runners/kinms_mock_pixelized.json

  python scripts/profile_phase1_regularization.py --mode eps --reg-center 1e5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup
from src.pipelines.phase1_likelihood import (
    format_likelihood_breakdown,
    phase1_likelihood_breakdown,
)
from src.pipelines.phase1_test import (
    adapt_images_for_pixelization,
    fixed_lens_galaxy_from_settings,
    interferometer_dataset_from_settings,
    pixelization_from_settings,
)
from src.pipelines.reconstruction import (
    reconstruction_mask_from_settings,
    use_positive_only_solver_from_settings,
)
from src.pipelines.runner import load_settings

ROOT = setup(__file__)

_SCAN_COLUMNS = (
    "reg",
    "chi_squared",
    "regularization_term",
    "log_likelihood_data",
    "log_likelihood_with_regularization",
    "figure_of_merit_log_evidence",
)


def _build_fit(settings, regularization_coefficient, use_jax=True):
    import autolens as al

    dataset = interferometer_dataset_from_settings(
        settings=settings,
        dataset_kind="moment0",
        transformer="nufft",
    )
    if use_jax:
        dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=False)

    mask_2d = reconstruction_mask_from_settings(settings)
    lens = fixed_lens_galaxy_from_settings(settings)
    mesh_type = settings["reconstruction"].get("mesh_type")
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

    return al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_positive_only_solver=use_positive_only_solver_from_settings(settings)
        ),
    )


def scan_regularization(settings, reg_values, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(
        "  "
        f"{'reg':>10}  {'chi2':>10}  {'reg_term':>10}  "
        f"{'logL_data':>12}  {'logL+reg':>12}  {'log_evidence':>12}"
    )
    for reg in reg_values:
        breakdown = _breakdown_for_reg(settings, reg)
        if breakdown is None:
            continue
        rows.append((reg, breakdown))
        print(
            f"  {reg:10.4e}  {breakdown['chi_squared']:10.2f}  "
            f"{breakdown['regularization_term']:10.2e}  "
            f"{breakdown['log_likelihood_data']:12.2f}  "
            f"{breakdown['log_likelihood_with_regularization']:12.2f}  "
            f"{breakdown['figure_of_merit_log_evidence']:12.2f}"
        )

    if not rows:
        raise RuntimeError("No regularization values produced a successful fit.")

    def _best_reg(key):
        values = [row[1][key] for row in rows]
        return rows[int(np.argmax(values))][0]

    best_data = _best_reg("log_likelihood_data")
    best_reg = _best_reg("log_likelihood_with_regularization")
    best_evidence = _best_reg("figure_of_merit_log_evidence")

    summary_path = output_dir / "regularization_scan.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Phase-1 regularization scan (fixed lens)\n"
            "LBFGS maximizes figure_of_merit_log_evidence, not log_likelihood_data.\n\n"
        )
        handle.write("\t".join(_SCAN_COLUMNS) + "\n")
        for reg, breakdown in rows:
            handle.write(
                f"{reg:.6e}\t{breakdown['chi_squared']:.6f}\t"
                f"{breakdown['regularization_term']:.6e}\t"
                f"{breakdown['log_likelihood_data']:.6f}\t"
                f"{breakdown['log_likelihood_with_regularization']:.6f}\t"
                f"{breakdown['figure_of_merit_log_evidence']:.6f}\n"
            )
        handle.write(f"\nbest_by_log_likelihood_data (overfits): {best_data:.6e}\n")
        handle.write(f"best_by_log_likelihood_with_regularization: {best_reg:.6e}\n")
        handle.write(f"best_by_log_evidence (LBFGS target): {best_evidence:.6e}\n")

    try:
        import matplotlib.pyplot as plt

        x = np.log10(reg_values)
        logl_data = [row[1]["log_likelihood_data"] for row in rows]
        logl_reg = [row[1]["log_likelihood_with_regularization"] for row in rows]
        log_ev = [row[1]["figure_of_merit_log_evidence"] for row in rows]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, logl_data, "o-", label="log_likelihood (data only)")
        ax.plot(x, logl_reg, "s-", label="log_likelihood + reg penalty")
        ax.plot(x, log_ev, "^-", label="log_evidence (LBFGS target)")
        ax.axvline(np.log10(best_evidence), color="C2", ls="--", alpha=0.6)
        ax.set_xlabel(r"$\log_{10}$(regularization coefficient)")
        ax.set_ylabel("figure of merit")
        ax.set_title("Phase-1 regularization profile (fixed lens)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "regularization_scan.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped ({exc})")

    print()
    print("Summary (which metric to trust):")
    print(f"  max log_likelihood_data (NO reg penalty, overfits): {best_data:.4e}")
    print(f"  max log_likelihood_with_regularization:           {best_reg:.4e}")
    print(f"  max log_evidence (LBFGS / figure_of_merit):       {best_evidence:.4e}")
    print(f"Wrote {summary_path}")
    return best_evidence


def _breakdown_for_reg(settings, reg):
    """Build fit and likelihood breakdown, or return None on failure."""
    try:
        return phase1_likelihood_breakdown(_build_fit(settings, reg))
    except Exception as exc:
        print(f"  WARNING: reg={reg:.4e} failed ({exc})")
        return None


def eps_sensitivity(settings, reg_center, rel_steps, output_dir):
    """
    Finite-difference gradient of figures of merit w.r.t. reg at fixed lens.

    Uses multiplicative perturbations ``reg * (1 ± frac)`` so step sizes scale
    with the coefficient magnitude (appropriate for reg ~ 1e5).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    breakdown_center = _breakdown_for_reg(settings, reg_center)
    if breakdown_center is None:
        raise RuntimeError(f"Central regularization {reg_center:.4e} failed to fit.")

    derivatives_data = []
    derivatives_evidence = []
    used_steps = []
    print(f"Central reg={reg_center:.4e}")
    print(format_likelihood_breakdown(breakdown_center))
    print("rel_step\treg_lo\treg_hi\tgrad_logL_data\tgrad_log_evidence")
    for frac in rel_steps:
        if frac <= 0.0 or frac >= 1.0:
            print(f"  WARNING: skipping invalid rel_step={frac}")
            continue
        reg_lo = reg_center * (1.0 - frac)
        reg_hi = reg_center * (1.0 + frac)
        lo = _breakdown_for_reg(settings, reg_lo)
        hi = _breakdown_for_reg(settings, reg_hi)
        if lo is None or hi is None:
            continue
        delta = reg_hi - reg_lo
        grad_data = (
            hi["log_likelihood_data"] - lo["log_likelihood_data"]
        ) / delta
        grad_ev = (
            hi["figure_of_merit_log_evidence"] - lo["figure_of_merit_log_evidence"]
        ) / delta
        derivatives_data.append(grad_data)
        derivatives_evidence.append(grad_ev)
        used_steps.append(frac)
        print(
            f"{frac:.4e}\t{reg_lo:.4e}\t{reg_hi:.4e}\t"
            f"{grad_data:.6e}\t{grad_ev:.6e}"
        )

    summary_path = output_dir / "eps_sensitivity.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"central_reg={reg_center}\n")
        handle.write(f"{format_likelihood_breakdown(breakdown_center)}\n")
        handle.write("rel_step\tgrad_logL_data\tgrad_log_evidence\n")
        for frac, gd, ge in zip(used_steps, derivatives_data, derivatives_evidence):
            handle.write(f"{frac}\t{gd}\t{ge}\n")

    print()
    print("Steps are multiplicative: reg_lo/hi = reg_center * (1 ± rel_step).")
    print("LBFGS maximizes log_evidence (figure_of_merit), not log_likelihood_data.")
    print("use_jax_gradient=true supplies analytical gradients for all parameters.")
    print(f"Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile phase-1 regularization figures of merit."
    )
    parser.add_argument(
        "--settings",
        default=str(ROOT / "settings" / "runners" / "kinms_mock_pixelized.json"),
    )
    parser.add_argument(
        "--mode",
        choices=("scan", "eps", "both"),
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "phase1_reg_profile"),
    )
    parser.add_argument(
        "--reg-center",
        type=float,
        default=1e5,
        help="Regularization for eps-sensitivity mode",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    output_dir = Path(args.output_dir)
    reg_values = np.logspace(3, 7, 17)

    print("Phase-1 regularization profile")
    print(f"  settings: {args.settings}")
    print(f"  output: {output_dir}")
    print("  NOTE: log_likelihood_data excludes the regularization penalty.")

    if args.mode in {"scan", "both"}:
        print("\n=== Regularization scan (fixed lens) ===")
        scan_regularization(settings, reg_values, output_dir)

    if args.mode in {"eps", "both"}:
        rel_steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
        print("\n=== Relative step sensitivity at fixed lens ===")
        eps_sensitivity(settings, args.reg_center, rel_steps, output_dir)


if __name__ == "__main__":
    main()
