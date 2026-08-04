#!/usr/bin/env python3
"""
Unlensed kinematic mock: isolate KinMS/GalPaK + UV path from gravitational lensing.

Reuses existing ALMA UV coverage / noise from a template dataprep product,
forward-models an unlensed cube (identity mass, θ_E=0) through the same
LensKin transformers used in fits, writes a new data directory, and evaluates
the truth model.

Supports ``model_name: "KinMS"`` (default) or ``"GalPak"`` via the settings
JSON / truth_parameters.

This skips CASA entirely — residuals are then due to KinMS cloudlet sampling
and/or injected noise, not lens ray-tracing or simobserve flux conventions.

**Noise is on by default.** Autolens pixelized source reconstructions need a
realistic noise floor; noiseless mocks make the inversion ill-behaved. Use
``--no-noise`` only for exact forward-model diagnostics.

Examples::

  # Mock with Gaussian noise from the template sigma map (default)
  python scripts/generate_unlensed_mock_and_diagnose.py

  # Noiseless mock (diagnostic only)
  python scripts/generate_unlensed_mock_and_diagnose.py --no-noise

  # Skip regenerate / write if data already exist
  python scripts/generate_unlensed_mock_and_diagnose.py --skip-generate
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

from scripts.generate_lensed_mock_and_diagnose import expand_template_spectral_axis
from scripts.test_truth_model import _visibility_fit_stats
from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines.cube_io import load_exported_array, write_exported_array
from src.pipelines.lens_model import lensing_enabled, validate_lensing_settings
from src.pipelines.pixelized_plots import save_cube, save_fit_triplet, save_image
from src.pipelines.runner import build_tracer, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import analysis_utils, autolens_utils, kinms_utils, plot_utils, spectral_utils


DEFAULT_SETTINGS = "settings/runners/kinms_mock_unlensed_parametric.json"
DEFAULT_TEMPLATE_DATA = "data/kinms_mock_pixelized"
DEFAULT_TEMPLATE_UID = "kinms_mock"


def _stem(prefix, uid, width):
    return f"{prefix}_{uid}_width_{width}_contsub"


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
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    model_name = settings.get("model_name", "KinMS")
    if model_name == "GalPak":
        # GalPaK only needs the source-plane grid for cube building / regridding.
        dataset_instance = type(
            "GalPaKGridInstance", (), {"grid_3d": source_grid_3d}
        )()
    else:
        dataset_instance = kinms_utils.make_instance_from_grid(
            grid_3d=source_grid_3d,
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


def generate_mock(
    settings,
    *,
    template_data,
    template_uid,
    add_noise=True,
    seed=0,
    pad_channels_each_side=None,
    noise_scale=1.0,
):
    settings = copy.deepcopy(settings)
    validate_lensing_settings(settings)
    out_dir = Path(settings["data_directory"])
    uid = settings["uids"][0]
    width = settings["width"]
    if pad_channels_each_side is None:
        pad_channels_each_side = int(
            settings.get("mock_pad_channels_each_side", 0) or 0
        )
    else:
        pad_channels_each_side = int(pad_channels_each_side)
    noise_scale = float(noise_scale)
    if not np.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError(f"noise_scale must be finite and >= 0; got {noise_scale!r}")

    print("\n=== Generate unlensed mock visibilities ===\n")
    print(f"  template = {template_data} (uid={template_uid})")
    print(f"  output   = {out_dir} (uid={uid})")
    print(f"  lensing.enabled = {lensing_enabled(settings)}")
    print(f"  θ_E      = {settings['lens_mass_model']['einstein_radius']}")
    print(f"  add_noise = {add_noise}")
    print(f"  noise_scale = {noise_scale}")
    print(f"  pad_channels_each_side = {pad_channels_each_side}")

    loaded = _copy_template_arrays(
        template_data,
        out_dir,
        template_uid,
        uid,
        width,
        pad_channels_each_side=pad_channels_each_side,
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

    # Freeze one kinematic realization for the mock sky (KinMS or GalPaK).
    source_cube = analysis.model_cube_from_instance(instance=instance)
    flip_y = bool(getattr(analysis, "_flip_kinms_y", True))
    image_grid = analysis.masked_dataset.grid_3d.grid_2d
    inst = analysis.masked_dataset.instance
    source_grid_2d = (
        inst.grid_3d.grid_2d
        if inst is not None and getattr(inst, "grid_3d", None) is not None
        else None
    )
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
        noise = rng.normal(size=model_vis.shape) * (sigma * noise_scale)
        data_vis = model_vis + noise
        print(
            f"  injected Gaussian noise (seed={seed}, amplitude={noise_scale}×σ)"
        )
        if noise_scale != 1.0:
            # Keep the exported noise map consistent with the injected amplitude.
            sigma_export = np.asarray(loaded["sigma_statwt"], dtype=float) * noise_scale
            sig_base = out_dir / _stem("sigma_statwt", uid, width)
            for old in sig_base.parent.glob(sig_base.name + ".*"):
                old.unlink()
            sig_path = write_exported_array(str(sig_base), sigma_export)
            print(f"  rewrote noise map ×{noise_scale}: {sig_path}")
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
        "noise_scale": noise_scale,
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
        else Path(settings["output_path"]) / "unlensed_truth_diagnostics"
    )
    out.mkdir(parents=True, exist_ok=True)

    dirty_data = autolens_utils.dirty_cube_from(
        visibilities=analysis.masked_dataset.data,
        transformers=analysis.transformers,
    )
    data_mom0 = dirty_data.sum(axis=0) * z_step

    print("\n=== Unlensed truth diagnostics ===\n")
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
        source_grid_2d=(
                analysis.masked_dataset.instance.grid_3d.grid_2d
                if analysis.masked_dataset.instance is not None
                and getattr(analysis.masked_dataset.instance, "grid_3d", None) is not None
                else None
            ),
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
        title="Unlensed mock dirty mom0",
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
        "--noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for injected noise (and exported σ); e.g. 1/3 for 3× quieter data",
    )
    parser.add_argument(
        "--pad-channels",
        type=int,
        default=None,
        help=(
            "Extra empty channels at each end of the spectral axis "
            "(overrides settings mock_pad_channels_each_side)"
        ),
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Use existing data_directory products; only run diagnostics",
    )
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    validate_lensing_settings(settings)
    frozen = None
    if not args.skip_generate:
        result = generate_mock(
            settings,
            template_data=args.template_data,
            template_uid=args.template_uid,
            add_noise=args.add_noise,
            seed=args.seed,
            pad_channels_each_side=args.pad_channels,
            noise_scale=args.noise_scale,
        )
        frozen = result["source_cube"]
    diagnose(settings, plots_dir=args.plots_dir, frozen_cube=frozen)


if __name__ == "__main__":
    main()
