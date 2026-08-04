#!/usr/bin/env python3
"""
Lensed mock + pixelized-SB truth diagnostics (no CASA).

Same mock generation as ``generate_lensed_mock_and_diagnose.py`` (parametric
KinMS → lens → NUFFT), then evaluates a **KinMSPixelized** model whose surface
brightness is fixed to the truth source cube (channel-mean map → inClouds) at
truth kinematics.

This isolates the KinMSPixelized forward model (cloudlet sampling of a fixed
SB map) from phase-1 reconstruction error.

Visibility noise σ is copied from the template dataprep ``sigma_statwt`` product
and used for χ² weights. **Noise is injected into the mock by default** —
Autolens pixelized source reconstructions need a realistic noise floor; use
``--no-noise`` only for exact forward-model checks. Residual plots include a
noise-normalised panel using a Monte-Carlo dirty-image σ from that same map.

Examples::

  # Generate noisy mock + pixelized truth residuals (default)
  python scripts/generate_lensed_mock_pixelized_and_diagnose.py

  # Noiseless mock (diagnostic only); reuse existing products
  python scripts/generate_lensed_mock_pixelized_and_diagnose.py --no-noise
  python scripts/generate_lensed_mock_pixelized_and_diagnose.py --skip-generate
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

from scripts.generate_lensed_mock_and_diagnose import (
    DEFAULT_TEMPLATE_DATA,
    DEFAULT_TEMPLATE_UID,
    _analysis_from_loaded,
    generate_mock,
)
from scripts.test_truth_model import _visibility_fit_stats
from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines.cube_io import load_exported_array
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils, kinms_utils, plot_utils, spectral_utils


DEFAULT_SETTINGS = "settings/runners/kinms_mock_lensed_pixelized.json"


def _frequencies_hz(frequencies):
    arr = np.squeeze(np.asarray(frequencies, dtype=float))
    if np.nanmax(arr) < 1.0e4:
        return arr * 1.0e9
    return arr


def truth_sb_map_from_source_cube(source_cube):
    """
    Channel-mean SB map in Jy/pixel/channel (same units as phase-1 maps).

    The frozen parametric KinMS cube is already Jy/pixel per channel.
    """
    return np.asarray(source_cube, dtype=float).mean(axis=0)


def _pixelized_analysis_from_loaded(
    settings,
    frequencies,
    uv_wavelengths,
    visibilities,
    sigma,
    sb_map,
    *,
    lens_centre=None,
):
    """Build Analysis with KinMSPixelized dataset instance from a fixed SB map."""
    settings = copy.deepcopy(settings)
    settings["model_name"] = "KinMSPixelized"

    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    autolens_utils.resolve_image_plane_grid_in_settings(settings, uv_wavelengths)
    img_n, img_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=img_n,
        pixel_scale=img_scale,
        n_channels=len(frequencies),
    )
    kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )

    flux_threshold = settings.get("reconstruction", {}).get("flux_threshold", 0.0)
    sb_input_units = settings.get("reconstruction", {}).get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    dataset_instance = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=kinms_grid_3d,
        z_step_kms=z_step_kms,
        sb_map=sb_map,
        flux_threshold=flux_threshold,
        sb_input_units=sb_input_units,
        **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
    )

    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
        frequencies_hz=_frequencies_hz(frequencies),
    )
    mask_3d = Mask3D.unmasked(
        n_channels=image_grid_3d.n_channels,
        shape_2d=image_grid_3d.shape_2d,
        pixel_scales=image_grid_3d.pixel_scales,
    )
    masked_dataset = MaskedDataset(
        dataset=dataset, mask_3d=mask_3d, instance=dataset_instance
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=masked_dataset.uv_wavelengths,
        mask_3d=masked_dataset.mask_3d,
        settings=settings,
    )
    tracer = build_tracer(settings, centre=lens_centre)
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )


def diagnose_pixelized(settings, *, plots_dir=None, frozen_cube=None):
    settings = copy.deepcopy(settings)
    settings["model_name"] = "KinMSPixelized"

    frequencies, uv, vis, sigma = load_cube_data(settings)
    z_step = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    extent = autolens_utils.image_extent_arcsec(settings["real_space_width"])
    n_chan = len(np.asarray(frequencies).reshape(-1))
    v_half = 0.5 * n_chan * z_step
    truth = settings.get("truth_parameters", {})
    v_proj = float(truth.get("maximum_velocity", 0.0)) * float(
        np.sin(np.radians(float(truth.get("inclination", 0.0))))
    )

    out = Path(
        plots_dir
        if plots_dir is not None
        else Path(settings["output_path"]) / "lensed_pixelized_truth_diagnostics"
    )
    out.mkdir(parents=True, exist_ok=True)

    if frozen_cube is None:
        frozen_path = Path(settings["data_directory"]) / "frozen_source_cube.npy"
        if not frozen_path.is_file():
            raise FileNotFoundError(
                f"Need frozen source cube at {frozen_path} "
                "(run without --skip-generate, or generate the lensed mock first)."
            )
        frozen_cube = np.load(frozen_path)

    sb_map = truth_sb_map_from_source_cube(frozen_cube)
    np.save(out / "truth_sb_map.npy", sb_map)
    save_image(
        sb_map,
        out / "truth_sb_map.png",
        title="Truth SB (channel mean, Jy/pix/chan)",
        extent=autolens_utils.image_extent_arcsec(
            settings.get("source_grid", {}).get(
                "real_space_width", settings["real_space_width"]
            )
        ),
    )

    # Ceiling: same frozen cube → lens → NUFFT (no pixelized re-sampling).
    parametric_analysis = _analysis_from_loaded(
        settings, frequencies, uv, vis, sigma
    )
    # Ensure parametric analysis uses KinMS for frozen-cube path only via
    # direct lensing of the cube (not model_data_from_instance).
    flip_y = bool(getattr(parametric_analysis, "_flip_kinms_y", True))
    lensed_frozen = analysis_utils.lensed_cube_from_tracer(
        cube=frozen_cube,
        tracer=parametric_analysis.tracer,
        grid=parametric_analysis.masked_dataset.grid_3d.grid_2d,
        z_mask=parametric_analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=parametric_analysis.masked_dataset.instance.grid_3d.grid_2d,
        output_shape=parametric_analysis.masked_dataset.grid_3d.shape_2d,
        flip_kinms_y=flip_y,
    )
    pb = getattr(parametric_analysis, "_primary_beam", None)
    frozen_vis = autolens_utils.visibilities_from_transformers_and_cube(
        cube=lensed_frozen,
        transformers=parametric_analysis.transformers,
        shape=parametric_analysis.masked_dataset.data.shape,
        z_mask=parametric_analysis.masked_dataset.z_mask,
        primary_beam=pb,
    )

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=parametric_analysis.masked_dataset.data,
        transformers=parametric_analysis.transformers,
    )
    data_mom0 = dirty_data.sum(axis=0) * z_step

    st_fr = _visibility_fit_stats(parametric_analysis, frozen_vis)
    dirty_fr = autolens_utils.dirty_cube_from(
        visibilities=frozen_vis, transformers=parametric_analysis.transformers
    )
    fr_mom0 = dirty_fr.sum(axis=0) * z_step

    print("\n=== Lensed pixelized truth diagnostics ===\n")
    print(f"  flip_kinms_y = {settings.get('flip_kinms_y_before_lensing')}")
    print(f"  θ_E = {settings['lens_mass_model']['einstein_radius']}")
    print(
        f"  n_chan={n_chan}, z_step={z_step:.3f} km/s, "
        f"half-width≈{v_half:.1f} km/s, v_proj_max≈{v_proj:.1f} km/s "
        f"(margin≈{v_half - v_proj:.1f} km/s)"
    )
    print(f"  plots -> {out}")
    print(
        f"  noise: template sigma_statwt (visibility weights); "
        f"mock data include injected noise unless generated with --no-noise"
    )
    print(
        f"  frozen cube ceiling:  chi2/N={st_fr['chi_squared_per_datum']:.6e}  "
        f"logL={st_fr['log_likelihood']:.2f}"
    )

    # Dirty-image σ from visibility σ (for residual / σ plots). Same transformers
    # as the data dirty cube.
    dirty_sigma = autolens_utils.dirty_noise_cube_from(
        noise_map=sigma,
        transformers=parametric_analysis.transformers,
        n_realizations=8,
        seed=0,
    )
    mom0_sigma = autolens_utils.dirty_mom0_noise_from_channel_noise(
        dirty_sigma, z_step
    )
    print(
        f"  dirty σ (MC): median channel rms={np.median(dirty_sigma):.4g}, "
        f"median mom0 σ={np.median(mom0_sigma):.4g}"
    )

    save_fit_triplet(
        data_mom0,
        fr_mom0,
        data_mom0 - fr_mom0,
        out / "dirty_mom0_frozen_cube.png",
        titles=("Data dirty mom0", "Frozen-cube model", "Data − frozen"),
        extent=extent,
        scale_mode="residual",
    )
    save_fit_triplet(
        data_mom0,
        fr_mom0,
        data_mom0 - fr_mom0,
        out / "dirty_mom0_frozen_cube_over_sigma.png",
        titles=("Data dirty mom0", "Frozen-cube model", "Data − frozen"),
        extent=extent,
        scale_mode="sigma",
        residual_sigma=mom0_sigma,
    )

    # Pixelized SB fixed to truth, kinematics at truth.
    analysis = _pixelized_analysis_from_loaded(
        settings, frequencies, uv, vis, sigma, sb_map
    )
    instance = truth_instance_from_settings(settings)
    pix_vis = analysis.model_data_from_instance(instance=instance)
    st_pix = _visibility_fit_stats(analysis, pix_vis)
    dirty_pix = autolens_utils.dirty_cube_from(
        visibilities=pix_vis, transformers=analysis.transformers
    )
    pix_mom0 = dirty_pix.sum(axis=0) * z_step
    print(
        f"  pixelized truth:  chi2/N={st_pix['chi_squared_per_datum']:.6f}  "
        f"logL={st_pix['log_likelihood']:.2f}  "
        f"cube_corr={np.corrcoef(dirty_data.ravel(), dirty_pix.ravel())[0, 1]:.4f}"
    )
    bulge = getattr(instance.galaxies.source, "bulge", instance.galaxies.source)
    pix_cube = bulge.profile_cube_from_masked_dataset(analysis.masked_dataset)
    ft = np.asarray(frozen_cube, dtype=float).sum(axis=(1, 2))
    pt = np.asarray(pix_cube, dtype=float).sum(axis=(1, 2))
    wing_fr = float((ft[:2].sum() + ft[-2:].sum()) / ft.sum())
    wing_pix = float((pt[:2].sum() + pt[-2:].sum()) / pt.sum())
    print(
        f"  spectrum wing frac frozen/pixelized: {wing_fr:.3f} / {wing_pix:.3f}"
    )
    print(
        f"  edge channel pix/frozen: ch0={pt[0] / ft[0]:.3f}  "
        f"ch-1={pt[-1] / ft[-1]:.3f}"
    )

    # Source-plane pixelized cube (pre-lensing): should look focussed like truth SB,
    # not an Einstein ring.
    kinms_grid = analysis.masked_dataset.instance.grid_3d
    coords = np.asarray(kinms_grid.grid_2d)
    if coords.ndim == 3 and coords.shape[-1] == 2:
        y = coords[:, :, 0]
        x = coords[:, :, 1]
    else:
        y = coords[:, 0]
        x = coords[:, 1]
    src_extent = autolens_utils.image_extent_from_bounding_box(
        [float(y.min()), float(y.max()), float(x.min()), float(x.max())]
    )
    pix_cube = np.asarray(pix_cube, dtype=float)
    np.save(out / "pixelized_source_cube.npy", pix_cube)
    pix_src_mean = pix_cube.mean(axis=0)
    pix_src_mom0 = pix_cube.sum(axis=0) * z_step
    truth_mom0 = sb_map * n_chan * z_step
    save_image(
        pix_src_mom0,
        out / "pixelized_source_mom0.png",
        title="Pixelized source-plane mom0 (Jy km/s/pix)",
        extent=src_extent,
    )
    save_image(
        pix_src_mean,
        out / "pixelized_source_sb_mean.png",
        title="Pixelized source-plane channel mean (Jy/pix/chan)",
        extent=src_extent,
    )
    save_fit_triplet(
        truth_mom0,
        pix_src_mom0,
        truth_mom0 - pix_src_mom0,
        out / "source_plane_truth_vs_pixelized.png",
        titles=("Truth source mom0", "Pixelized source mom0", "Truth − pixelized"),
        extent=src_extent,
        scale_mode="residual",
    )
    save_cube(
        pix_cube,
        out / "pixelized_source_channels.png",
        extent=src_extent,
    )
    # Image-plane lensed cube (Jy/pixel, before NUFFT) for focus vs ring check.
    flip_y_pix = bool(getattr(analysis, "_flip_kinms_y", True))
    lensed_pix = analysis_utils.lensed_cube_from_tracer(
        cube=pix_cube,
        tracer=analysis.tracer,
        grid=analysis.masked_dataset.grid_3d.grid_2d,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=kinms_grid.grid_2d,
        output_shape=analysis.masked_dataset.grid_3d.shape_2d,
        flip_kinms_y=flip_y_pix,
    )
    lensed_pix_mom0 = np.asarray(lensed_pix, dtype=float).sum(axis=0) * z_step
    save_image(
        lensed_pix_mom0,
        out / "pixelized_lensed_mom0.png",
        title="Pixelized lensed mom0 (Jy km/s/pix, pre-NUFFT)",
        extent=extent,
    )
    print(
        f"  source-plane plots: pixelized_source_mom0.png, "
        f"source_plane_truth_vs_pixelized.png "
        f"(src extent={src_extent})"
    )

    save_fit_triplet(
        data_mom0,
        pix_mom0,
        data_mom0 - pix_mom0,
        out / "dirty_mom0_pixelized_truth.png",
        titles=("Data dirty mom0", "Pixelized truth SB", "Data − pixelized"),
        extent=extent,
        scale_mode="residual",
    )
    save_fit_triplet(
        data_mom0,
        pix_mom0,
        data_mom0 - pix_mom0,
        out / "dirty_mom0_pixelized_truth_over_sigma.png",
        titles=("Data dirty mom0", "Pixelized truth SB", "Data − pixelized"),
        extent=extent,
        scale_mode="sigma",
        residual_sigma=mom0_sigma,
    )

    resid = dirty_data - dirty_pix
    lim = float(np.nanmax(np.abs(resid))) or 1.0
    fig, _ = plot_utils.plot_cube(
        cube=resid, ncols=8, vmin=-lim, vmax=lim, cmap="RdBu_r"
    )
    fig.suptitle(
        "Channel residuals (pixelized truth SB — cloudlet / SB sampling floor)",
        y=1.02,
    )
    fig.savefig(out / "channel_residuals_pixelized_truth.png", bbox_inches="tight", dpi=130)
    plt.close(fig)

    safe_sigma = np.where(
        np.isfinite(dirty_sigma) & (dirty_sigma > 0.0), dirty_sigma, np.nan
    )
    resid_over_sigma = resid / safe_sigma
    finite = resid_over_sigma[np.isfinite(resid_over_sigma)]
    rmin = float(np.min(finite)) if finite.size else float("nan")
    rmax = float(np.max(finite)) if finite.size else float("nan")
    peak = float(np.nanmax(np.abs(resid_over_sigma))) if resid_over_sigma.size else 0.0
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
        f"Channel residuals / σ_dirty (colour ±5σ; "
        f"map min={rmin:.2f}σ, max={rmax:.2f}σ, peak|r|={peak:.2f}σ)",
        y=1.02,
    )
    fig.savefig(
        out / "channel_residuals_pixelized_truth_over_sigma.png",
        bbox_inches="tight",
        dpi=130,
    )
    plt.close(fig)
    np.save(out / "dirty_sigma_cube.npy", dirty_sigma)
    np.save(out / "dirty_mom0_sigma.npy", mom0_sigma)

    save_cube(dirty_data, out / "dirty_channels_data.png", extent=extent)
    save_cube(dirty_pix, out / "dirty_channels_model.png", extent=extent)
    save_image(
        data_mom0,
        out / "dirty_mom0_data.png",
        title="Lensed mock dirty mom0",
        extent=extent,
    )
    print(f"\n  Done. Inspect plots under {out}")
    return {
        "frozen": st_fr,
        "pixelized_truth": st_pix,
        "spectrum": {
            "wing_frac_frozen": wing_fr,
            "wing_frac_pixelized": wing_pix,
            "edge_ratio_ch0": float(pt[0] / ft[0]),
            "edge_ratio_ch_last": float(pt[-1] / ft[-1]),
            "pix_over_frozen": (pt / ft).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--template-data", default=DEFAULT_TEMPLATE_DATA)
    parser.add_argument("--template-uid", default=DEFAULT_TEMPLATE_UID)
    parser.set_defaults(add_noise=True)
    parser.add_argument(
        "--add-noise",
        action="store_true",
        dest="add_noise",
        help="Inject Gaussian noise from the template sigma map (default)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_false",
        dest="add_noise",
        help=(
            "Noiseless mock (data = model). Diagnostic only — Autolens "
            "pixelized source solutions struggle without a noise floor"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Use existing data_directory products; only run diagnostics",
    )
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    # Mock generation uses parametric KinMS (intensity / re in truth_parameters).
    gen_settings = copy.deepcopy(settings)
    gen_settings["model_name"] = "KinMS"
    if "intensity" not in gen_settings.get("truth_parameters", {}):
        # Fall back to parametric lensed defaults if pixelized settings omit SB params.
        gen_settings.setdefault("truth_parameters", {})
        gen_settings["truth_parameters"].setdefault("intensity", 12.5)
        gen_settings["truth_parameters"].setdefault("effective_radius", 0.5)

    frozen = None
    if not args.skip_generate:
        result = generate_mock(
            gen_settings,
            template_data=args.template_data,
            template_uid=args.template_uid,
            add_noise=args.add_noise,
            seed=args.seed,
        )
        frozen = result["source_cube"]

    diagnose_pixelized(
        settings,
        plots_dir=args.plots_dir,
        frozen_cube=frozen,
    )


if __name__ == "__main__":
    main()
