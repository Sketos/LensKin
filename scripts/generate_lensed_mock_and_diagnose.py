#!/usr/bin/env python3
"""
Lensed kinematic mock: KinMS + gravitational lensing + UV path (no CASA).

Reuses existing ALMA UV coverage / noise from a template dataprep product,
forward-models a KinMS cube through the LensKin lens mass model and the same
transformers used in fits, writes a new data directory, and evaluates the
truth model.

This skips CASA entirely — same pipeline as generate_unlensed_mock_and_diagnose.py
but with a non-zero Einstein radius / shear (full lensing step).

**Noise is on by default.** Autolens pixelized source reconstructions need a
realistic noise floor; noiseless mocks make the inversion ill-behaved. Use
``--no-noise`` only for exact forward-model diagnostics.

Examples::

  # Mock with Gaussian noise from the template sigma map (default)
  python scripts/generate_lensed_mock_and_diagnose.py

  # Noiseless mock (diagnostic only)
  python scripts/generate_lensed_mock_and_diagnose.py --no-noise

  # Skip regenerate / write if data already exist
  python scripts/generate_lensed_mock_and_diagnose.py --skip-generate
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

from scripts.test_truth_model import _visibility_fit_stats
from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines.cube_io import load_exported_array, write_exported_array
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import build_tracer, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils, kinms_utils, plot_utils, spectral_utils


DEFAULT_SETTINGS = "settings/runners/kinms_mock_lensed_parametric.json"
DEFAULT_TEMPLATE_DATA = "data/kinms_mock_pixelized"
DEFAULT_TEMPLATE_UID = "kinms_mock"


def _stem(prefix, uid, width):
    return f"{prefix}_{uid}_width_{width}_contsub"


def expand_template_spectral_axis(
    frequencies,
    uv_wavelengths,
    sigma_statwt,
    *,
    pad_channels_each_side: int,
):
    """
    Extend the spectral axis by ``pad_channels_each_side`` on each end.

    Frequencies continue the template grid with constant channel spacing.
    UV coordinates are scaled ∝ frequency from the nearest template channel
    (template UV already scales exactly with frequency). Sigma is copied from
    the nearest edge channel (fine for noiseless mocks / diagnostic noise).
    """
    pad = int(pad_channels_each_side)
    if pad < 0:
        raise ValueError(f"pad_channels_each_side must be >= 0; got {pad}")
    if pad == 0:
        return (
            np.asarray(frequencies),
            np.asarray(uv_wavelengths),
            np.asarray(sigma_statwt),
        )

    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    uv_wavelengths = np.asarray(uv_wavelengths, dtype=float)
    sigma_statwt = np.asarray(sigma_statwt, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError(f"Expected 1D frequencies; got shape {frequencies.shape}")
    if uv_wavelengths.ndim != 3:
        raise ValueError(
            f"Expected uv_wavelengths (n_chan, n_vis, 2); got {uv_wavelengths.shape}"
        )
    if sigma_statwt.ndim != 4:
        raise ValueError(
            f"Expected sigma_statwt (n_pol, n_chan, n_vis, 2); got {sigma_statwt.shape}"
        )
    n_chan = frequencies.size
    if uv_wavelengths.shape[0] != n_chan or sigma_statwt.shape[1] != n_chan:
        raise ValueError(
            "Channel-axis mismatch among frequencies / uv_wavelengths / sigma_statwt "
            f"({n_chan}, {uv_wavelengths.shape[0]}, {sigma_statwt.shape[1]})"
        )

    df = float(np.mean(np.diff(frequencies)))
    if not np.isfinite(df) or df == 0.0:
        raise ValueError(f"Invalid frequency channel spacing df={df}")

    f_lo = frequencies[0] - df * np.arange(pad, 0, -1)
    f_hi = frequencies[-1] + df * np.arange(1, pad + 1)
    frequencies_out = np.concatenate([f_lo, frequencies, f_hi])

    def _scale_uv(uv_ref, f_ref, f_new):
        return uv_ref * (float(f_new) / float(f_ref))

    uv_lo = np.stack(
        [_scale_uv(uv_wavelengths[0], frequencies[0], f) for f in f_lo],
        axis=0,
    )
    uv_hi = np.stack(
        [_scale_uv(uv_wavelengths[-1], frequencies[-1], f) for f in f_hi],
        axis=0,
    )
    uv_out = np.concatenate([uv_lo, uv_wavelengths, uv_hi], axis=0)

    sig_lo = np.repeat(sigma_statwt[:, :1], pad, axis=1)
    sig_hi = np.repeat(sigma_statwt[:, -1:], pad, axis=1)
    sigma_out = np.concatenate([sig_lo, sigma_statwt, sig_hi], axis=1)

    return frequencies_out, uv_out, sigma_out


def _copy_template_arrays(
    template_dir,
    out_dir,
    template_uid,
    uid,
    width,
    *,
    pad_channels_each_side: int = 0,
):
    """Copy frequencies / uv / sigma from template; visibilities written later."""
    template_dir = Path(template_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "frequencies": "frequencies",
        "uv_wavelengths": "uv_wavelengths",
        "sigma_statwt": "sigma_statwt",
    }
    loaded = {}
    for key, prefix in mapping.items():
        src = template_dir / _stem(prefix, template_uid, width)
        data = load_exported_array(src)
        loaded[key] = np.asarray(data)

    if pad_channels_each_side:
        loaded["frequencies"], loaded["uv_wavelengths"], loaded["sigma_statwt"] = (
            expand_template_spectral_axis(
                loaded["frequencies"],
                loaded["uv_wavelengths"],
                loaded["sigma_statwt"],
                pad_channels_each_side=pad_channels_each_side,
            )
        )
        n_chan = len(np.asarray(loaded["frequencies"]).reshape(-1))
        z_step = spectral_utils.z_step_kms_from_data_frequencies(loaded["frequencies"])
        print(
            f"  padded spectral axis by ±{int(pad_channels_each_side)} channels "
            f"-> n_chan={n_chan}, half-width≈{0.5 * n_chan * z_step:.1f} km/s"
        )

    for key, prefix in mapping.items():
        dest_base = out_dir / _stem(prefix, uid, width)
        # Drop any previous extension; write_exported_array adds .fits/.npy.
        for old in dest_base.parent.glob(dest_base.name + ".*"):
            old.unlink()
        path = write_exported_array(str(dest_base), loaded[key])
        print(f"  wrote {path}")
    return loaded


def _split_pols_for_export(model_vis):
    """
    Analysis visibilities are (n_chan, 2*n_vis, 2) after pol concatenation.
    Dataprep products are (2, n_chan, n_vis, 2).
    """
    model_vis = np.asarray(model_vis)
    if model_vis.ndim != 3 or model_vis.shape[-1] != 2:
        raise ValueError(f"Unexpected model visibility shape {model_vis.shape}")
    n_chan, n_vis_tot, _ = model_vis.shape
    if n_vis_tot % 2 != 0:
        raise ValueError(f"Visibility count {n_vis_tot} is not divisible by 2")
    n_vis = n_vis_tot // 2
    return np.stack(
        [model_vis[:, :n_vis, :], model_vis[:, n_vis:, :]],
        axis=0,
    )


def _analysis_from_loaded(settings, frequencies, uv_wavelengths, visibilities, sigma):
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
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    dataset_instance = kinms_utils.make_instance_from_grid(
        grid_3d=kinms_grid_3d,
        z_step_kms=z_step_kms,
        attach_grid=True,
        disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
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
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=build_tracer(settings),
        settings=settings,
    )


def generate_mock(settings, *, template_data, template_uid, add_noise=True, seed=0):
    settings = copy.deepcopy(settings)
    out_dir = Path(settings["data_directory"])
    uid = settings["uids"][0]
    width = settings["width"]

    print("\n=== Generate lensed mock visibilities ===\n")
    print(f"  template = {template_data} (uid={template_uid})")
    print(f"  output   = {out_dir} (uid={uid})")
    print(f"  θ_E      = {settings['lens_mass_model']['einstein_radius']}")
    print(f"  add_noise = {add_noise}")

    pad = int(settings.get("mock_pad_channels_each_side", 0) or 0)
    loaded = _copy_template_arrays(
        template_data,
        out_dir,
        template_uid,
        uid,
        width,
        pad_channels_each_side=pad,
    )
    # Build concatenated arrays as load_cube_data would.
    frequencies = loaded["frequencies"]
    uv = np.concatenate((loaded["uv_wavelengths"], loaded["uv_wavelengths"]), axis=1)
    sigma = np.concatenate((loaded["sigma_statwt"][0], loaded["sigma_statwt"][1]), axis=1)
    # Placeholder visibilities (replaced after model prediction).
    n_chan = len(frequencies)
    n_vis = loaded["uv_wavelengths"].shape[1]
    vis_placeholder = np.zeros((n_chan, 2 * n_vis, 2), dtype=float)

    analysis = _analysis_from_loaded(
        settings, frequencies, uv, vis_placeholder, sigma
    )
    instance = truth_instance_from_settings(settings)

    # Freeze one KinMS realization for the mock sky.
    source_cube = analysis.model_cube_from_instance(instance=instance)
    flip_y = bool(getattr(analysis, "_flip_kinms_y", True))
    image_grid = analysis.masked_dataset.grid_3d.grid_2d
    source_grid_2d = analysis.masked_dataset.instance.grid_3d.grid_2d
    lensed_cube = analysis_utils.lensed_cube_from_tracer(
        cube=source_cube,
        tracer=analysis.tracer,
        grid=image_grid,
        z_mask=analysis.masked_dataset.mask_3d.z_mask,
        source_grid_2d=source_grid_2d,
        output_shape=analysis.masked_dataset.grid_3d.shape_2d,
        flip_kinms_y=flip_y,
    )
    model_vis = autolens_utils.visibilities_from_transformers_and_cube(
        cube=lensed_cube,
        transformers=analysis.transformers,
        shape=analysis.masked_dataset.data.shape,
        z_mask=analysis.masked_dataset.z_mask,
    )

    if add_noise:
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=model_vis.shape) * sigma
        data_vis = model_vis + noise
        print(f"  injected Gaussian noise (seed={seed})")
    else:
        data_vis = model_vis
        print(
            "  noiseless mock (data = model visibilities); "
            "Autolens pixelizations typically need noise — prefer the default"
        )

    export = _split_pols_for_export(data_vis)
    vis_base = out_dir / _stem("visibilities", uid, width)
    for old in vis_base.parent.glob(vis_base.name + ".*"):
        old.unlink()
    vis_path = write_exported_array(str(vis_base), export)
    print(f"  wrote {vis_path}")

    # Save frozen source cube for perfect-recovery check.
    cube_path = out_dir / "frozen_source_cube.npy"
    np.save(cube_path, source_cube)
    print(f"  wrote {cube_path}")

    return {
        "settings": settings,
        "source_cube": source_cube,
        "lensed_cube": lensed_cube,
        "model_vis": model_vis,
        "data_vis": data_vis,
        "analysis": analysis,
        "instance": instance,
    }


def diagnose(settings, *, plots_dir=None, frozen_cube=None):
    from src.pipelines.runner import load_cube_data

    settings = copy.deepcopy(settings)
    frequencies, uv, vis, sigma = load_cube_data(settings)
    analysis = _analysis_from_loaded(settings, frequencies, uv, vis, sigma)
    instance = truth_instance_from_settings(settings)
    z_step = float(analysis.masked_dataset.z_step_kms)
    extent = autolens_utils.image_extent_arcsec(settings["real_space_width"])

    out = Path(
        plots_dir
        if plots_dir is not None
        else Path(settings["output_path"]) / "lensed_truth_diagnostics"
    )
    out.mkdir(parents=True, exist_ok=True)

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    data_mom0 = dirty_data.sum(axis=0) * z_step

    print("\n=== Lensed truth diagnostics ===\n")
    print(f"  flip_kinms_y = {settings.get('flip_kinms_y_before_lensing')}")
    print(f"  flip_velocity = {settings.get('flip_velocity_axis')}")
    print(f"  θ_E = {settings['lens_mass_model']['einstein_radius']}")
    print(f"  plots -> {out}")

    # 1) Frozen-cube recovery (same sky as mock generation if provided)
    if frozen_cube is None:
        frozen_path = Path(settings["data_directory"]) / "frozen_source_cube.npy"
        if frozen_path.is_file():
            frozen_cube = np.load(frozen_path)

    if frozen_cube is not None:
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
        frozen_vis = autolens_utils.visibilities_from_transformers_and_cube(
            cube=lensed,
            transformers=analysis.transformers,
            shape=analysis.masked_dataset.data.shape,
            z_mask=analysis.masked_dataset.z_mask,
        )
        st_fr = _visibility_fit_stats(analysis, frozen_vis)
        dirty_fr = autolens_utils.dirty_cube_from(
            visibilities=frozen_vis, transformers=analysis.transformers
        )
        fr_mom0 = dirty_fr.sum(axis=0) * z_step
        print(
            f"  frozen cube:  chi2/N={st_fr['chi_squared_per_datum']:.6e}  "
            f"logL={st_fr['log_likelihood']:.2f}"
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
        resid_fr = dirty_data - dirty_fr
        lim = float(np.nanmax(np.abs(resid_fr))) or 1.0
        fig, _ = plot_utils.plot_cube(
            cube=resid_fr, ncols=8, vmin=-lim, vmax=lim, cmap="RdBu_r"
        )
        fig.suptitle("Channel residuals (frozen cube — should be ~0 if noiseless)", y=1.02)
        fig.savefig(out / "channel_residuals_frozen.png", bbox_inches="tight", dpi=130)
        plt.close(fig)

    # 2) Regenerated KinMS at truth (cloudlet Monte Carlo)
    regen_vis = analysis.model_data_from_instance(instance=instance)
    st_rg = _visibility_fit_stats(analysis, regen_vis)
    dirty_rg = autolens_utils.dirty_cube_from(
        visibilities=regen_vis, transformers=analysis.transformers
    )
    rg_mom0 = dirty_rg.sum(axis=0) * z_step
    print(
        f"  regenerated:  chi2/N={st_rg['chi_squared_per_datum']:.6f}  "
        f"logL={st_rg['log_likelihood']:.2f}  "
        f"cube_corr={np.corrcoef(dirty_data.ravel(), dirty_rg.ravel())[0,1]:.4f}"
    )
    save_fit_triplet(
        data_mom0,
        rg_mom0,
        data_mom0 - rg_mom0,
        out / "dirty_mom0_regenerated_kinms.png",
        titles=("Data dirty mom0", "Regenerated KinMS", "Data − regen"),
        extent=extent,
        scale_mode="residual",
    )
    resid_rg = dirty_data - dirty_rg
    lim = float(np.nanmax(np.abs(resid_rg))) or 1.0
    fig, _ = plot_utils.plot_cube(
        cube=resid_rg, ncols=8, vmin=-lim, vmax=lim, cmap="RdBu_r"
    )
    fig.suptitle("Channel residuals (regenerated KinMS — cloudlet noise floor)", y=1.02)
    fig.savefig(out / "channel_residuals_regenerated.png", bbox_inches="tight", dpi=130)
    plt.close(fig)

    save_cube(dirty_data, out / "dirty_channels_data.png", extent=extent)
    save_cube(dirty_rg, out / "dirty_channels_model.png", extent=extent)
    save_image(
        data_mom0,
        out / "dirty_mom0_data.png",
        title="Lensed mock dirty mom0",
        extent=extent,
    )
    print(f"\n  Done. Inspect plots under {out}")
    return {"frozen": st_fr if frozen_cube is not None else None, "regenerated": st_rg}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--template-data",
        default=DEFAULT_TEMPLATE_DATA,
        help="Existing dataprep dir providing UV / frequencies / sigma",
    )
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
    frozen = None
    if not args.skip_generate:
        result = generate_mock(
            settings,
            template_data=args.template_data,
            template_uid=args.template_uid,
            add_noise=args.add_noise,
            seed=args.seed,
        )
        frozen = result["source_cube"]
    diagnose(settings, plots_dir=args.plots_dir, frozen_cube=frozen)


if __name__ == "__main__":
    main()
