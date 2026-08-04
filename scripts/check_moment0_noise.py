#!/usr/bin/env python
"""
Diagnose moment-0 dataset construction for phase-1 pixelized fits.

Compares per-channel statwt sigma, collapsed moment-0 arrays, empirical channel
scatter, and implied median signal-to-noise.
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
from src.pipelines.moment0_noise import (
    channel_scatter_around_mean,
    moment0_arrays_from,
    moment0_settings_from_reconstruction,
)
from src.pipelines.runner import load_cube_data, load_settings

ROOT = setup(__file__)


def main():
    parser = argparse.ArgumentParser(description="Check moment-0 dataset construction.")
    parser.add_argument(
        "--settings",
        default=str(ROOT / "settings" / "runners" / "kinms_mock_pixelized.json"),
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    _, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    rec_cfg = settings["reconstruction"]
    moment0_cfg = moment0_settings_from_reconstruction(rec_cfg)

    vis_c = visibilities[..., 0] + 1j * visibilities[..., 1]
    n_channels = vis_c.shape[0]
    n_vis = vis_c.shape[1]
    ref = n_channels // 2

    scatter = channel_scatter_around_mean(visibilities)
    vis_mom0, sigma_mom0, uv_mom0 = moment0_arrays_from(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        moment0_settings=moment0_cfg,
    )

    sigma_per_ch = np.mean(sigma[..., 0], axis=-1)
    sigma_ref = sigma[ref, ..., 0]

    print("Moment-0 dataset diagnostic")
    print(f"  settings: {args.settings}")
    print(f"  n_channels: {n_channels}")
    print(f"  n_vis per channel: {n_vis}")
    print(f"  moment0 config: {moment0_cfg}")
    print()
    print("Per-channel statwt sigma (median over visibilities):")
    print(f"  all channels identical: {np.allclose(sigma_per_ch, sigma_per_ch[0])}")
    print(f"  median sigma: {np.median(sigma[..., 0]):.4e}")
    print()
    print("Collapsed moment-0 (current settings):")
    print(f"  collapse_mode: {moment0_cfg['collapse_mode']}")
    print(f"  n_vis total: {vis_mom0.shape[0]}")
    print(f"  sigma median: {np.median(sigma_mom0[..., 0]):.4e}")
    print(f"  data amp median: {np.median(np.abs(vis_mom0)):.4e}")
    print(
        "  implied S/N median: "
        f"{np.median(np.abs(vis_mom0) / np.mean(sigma_mom0, axis=-1)):.2f}"
    )
    if moment0_cfg["collapse_mode"] == "concatenate":
        print(
            "  expected n_vis: "
            f"{n_channels * n_vis} "
            f"(matches: {vis_mom0.shape[0] == n_channels * n_vis})"
        )
        print(f"  uv shape: {uv_mom0.shape}")
    print()
    print("Reference single channel:")
    print(f"  channel index: {ref}")
    print(f"  sigma median: {np.median(sigma_ref):.4e}")
    print(f"  data amp median: {np.median(np.abs(vis_c[ref])):.4e}")
    print(
        "  implied S/N median: "
        f"{np.median(np.abs(vis_c[ref]) / sigma_ref):.2f}"
    )
    print()
    print("Channel scatter around mean (kinematic structure, mean mode only):")
    print(f"  median scatter: {np.median(scatter):.4e}")
    if moment0_cfg["collapse_mode"] == "mean":
        print(
            "  scatter / moment0 sigma: "
            f"{np.median(scatter / sigma_mom0[..., 0]):.2f}"
        )
        expected_mom0_sigma = np.median(sigma[..., 0]) / np.sqrt(n_channels)
        print()
        print("Checks:")
        print(
            "  independent_mean formula gives sigma/sqrt(N): "
            f"{expected_mom0_sigma:.4e} "
            f"(matches collapsed: {np.isclose(np.median(sigma_mom0[..., 0]), expected_mom0_sigma * moment0_cfg['sigma_scale'])})"
        )
        if np.median(scatter) > 3 * np.median(sigma_mom0[..., 0]):
            print(
                "  NOTE: channel scatter >> moment-0 noise. Try "
                "collapse_mode: concatenate or --dataset channel."
            )
    else:
        print("  (not applicable for concatenate mode)")


if __name__ == "__main__":
    main()
