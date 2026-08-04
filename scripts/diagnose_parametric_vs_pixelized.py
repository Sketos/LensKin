#!/usr/bin/env python3
"""
Compare parametric vs pixelized phase-2 KinMS setups on the same mock data.

Parametric-only sanity check (no phase 1):
  python scripts/diagnose_parametric_vs_pixelized.py \\
    --settings settings/runners/kinms_mock_parametric.json

Pixelized vs parametric comparison (runs phase 1; slow):
  python scripts/diagnose_parametric_vs_pixelized.py \\
    --settings settings/runners/kinms_mock_pixelized.json \\
    --run-phase1
"""
import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup

setup(__file__)

import autofit as af
import numpy as np
from astropy import units

from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.model import profiles
from src.pipelines import reconstruction
from src.pipelines.priors import source_model_from_profile
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils

# Autofit vector order for kinMS (see model.info): z, I, re, phi, inc, vmax, sigma, rt, c0, c1
TRUTH_KINMS_VECTOR = [0.0, 0.05, 0.05, 90.0, 65.0, 320.0, 35.0, 0.05, 0.0, 0.0]


def _spectral_setup(settings, frequencies):
    return spectral_utils.z_step_kms_from_data_frequencies(frequencies)


def _build_analysis(settings, dataset_instance, lens_centre, grid_3d_for_kinms):
    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    z_step_kms = _spectral_setup(settings, frequencies)

    n_pixels, pixel_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=len(frequencies),
    )
    mask_3d = Mask3D.unmasked(
        n_channels=image_grid_3d.n_channels,
        shape_2d=image_grid_3d.shape_2d,
        pixel_scales=image_grid_3d.pixel_scales,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
        settings=settings,
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    tracer = build_tracer(settings, centre=lens_centre)
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
        instance=dataset_instance,
    )
    return analysis_mod.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
    ), masked_dataset


def _log_likelihood(analysis_instance, model, vector):
    instance = model.instance_from_vector(vector=vector)
    try:
        return float(analysis_instance.log_likelihood_function(instance=instance))
    except af.exc.FitException:
        # The Analysis raises FitException when likelihood is NaN. Print enough
        # context to diagnose parameter-order / invalid-model issues.
        print("\n[diagnose] FitException (NaN likelihood). Instance was:\n")
        try:
            print(instance)
        except Exception:
            print(repr(instance))

        try:
            src = instance.galaxies.source
            print("\n[diagnose] source parameters:\n")
            for k in sorted(getattr(src, "__dict__", {}).keys()):
                if k.startswith("_"):
                    continue
                print(f"  {k} = {getattr(src, k)!r}")
        except Exception as exc:
            print(f"\n[diagnose] could not introspect source params: {exc!r}")

        try:
            cube = analysis_instance.model_cube_from_instance(instance=instance)
            finite_frac = float(np.isfinite(cube).mean())
            print(f"\n[diagnose] model cube finite fraction = {finite_frac:.6f}")
            print(
                f"[diagnose] model cube min/max = "
                f"{np.nanmin(cube):.3g} / {np.nanmax(cube):.3g}"
            )
        except Exception as exc:
            print(f"\n[diagnose] model cube generation failed: {exc!r}")

        try:
            model_data = analysis_instance.model_data_from_instance(instance=instance)
            finite_frac = float(np.isfinite(model_data).mean())
            rms = float(np.sqrt(np.nanmean(np.abs(model_data) ** 2)))
            print(f"[diagnose] model vis finite fraction = {finite_frac:.6f}")
            print(f"[diagnose] model vis rms = {rms:.3g}")
        except Exception as exc:
            print(f"[diagnose] model vis generation failed: {exc!r}")

        raise


def _moment0_dirty_rms(analysis_instance, model, vector):
    instance = model.instance_from_vector(vector=vector)
    model_data = analysis_instance.model_data_from_instance(instance=instance)
    data = analysis_instance.masked_dataset.data
    mask = analysis_instance.masked_dataset.uv_mask == False
    residual = data[mask] - model_data[mask]
    return float(np.sqrt(np.mean(residual**2)))

def _pixelized_truth_vector_from_settings(settings, n):
    """
    Convert the pixelized ``debug_instance_vector`` into the Autofit vector order.

    Pixelized debug vectors are stored in settings as:
      [z, centre_0, centre_1, phi, inclination, turnover_radius,
       maximum_velocity, velocity_dispersion, vmax_black_hole]

    The model vector order produced by ``source_model_from_profile`` places the
    tuple prior (centre) last, and preserves insertion order for scalar kwargs.
    For the default JSON ordering in ``kinms_mock_pixelized.json`` this yields:

      [z_centre, phi, inclination, maximum_velocity, velocity_dispersion,
       turnover_radius, centre_0, centre_1]
    """
    truth = list(settings.get("debug_instance_vector", []))
    if len(truth) < 8:
        raise ValueError("Pixelized debug_instance_vector must have >= 8 entries.")

    # If vmax_black_hole is present (len 9), drop it (it's fixed in settings).
    if len(truth) >= 9:
        truth = truth[:9]

    z, c0, c1, phi, inc, rt, vmax, sigma = truth[:8]
    mapped = [z, phi, inc, vmax, sigma, rt, c1, c0]
    if len(mapped) != n:
        mapped = mapped[:n]
    return mapped

def _truth_vector_for_model(settings, model, fallback_vector):
    """
    Return a truth vector compatible with the model's prior count.

    Settings JSONs sometimes include fixed parameters (e.g. ``vmax_black_hole``)
    in ``debug_instance_vector`` even though they are not part of the sampled
    parameter vector. Autofit enforces that the vector length equals the
    model's prior count, so we trim any extra trailing values.
    """
    truth = settings.get("debug_instance_vector", fallback_vector)
    truth = list(truth)
    n = model.prior_count

    # Special case: pixelized settings store a debug vector in a different order
    # (see ``_pixelized_truth_vector_from_settings``). If we just trim the first
    # 8 values we can end up feeding vmax=0.05, sigma=320, rt=35, etc → NaNs.
    if settings.get("model_name") == "KinMSPixelized" and n == 8:
        return _pixelized_truth_vector_from_settings(settings, n=n)

    if len(truth) < n:
        fb = list(fallback_vector)
        if len(fb) >= n:
            truth = fb
        else:
            raise ValueError(
                f"debug_instance_vector too short for model: {len(truth)} < {n}"
            )
    if len(truth) != n:
        truth = truth[:n]
    return truth


def _truth_kinms_vector(settings):
    return settings.get("debug_instance_vector", TRUTH_KINMS_VECTOR)


def _kinms_parametric_model(settings):
    kinms_priors = dict(settings["priors"])
    kinms_priors["intensity"] = {
        "type": "LogUniformPrior",
        "lower_limit": 0.01,
        "upper_limit": 0.15,
    }
    kinms_priors["effective_radius"] = {
        "type": "LogUniformPrior",
        "lower_limit": 0.03,
        "upper_limit": 0.07,
    }
    return af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(profiles.kinMS, kinms_priors)
        )
    )


def _run_parametric_sanity_check(settings, frequencies, z_step_kms):
    truth_vector = _truth_kinms_vector(settings)
    img_n, img_pix, img_width = autolens_utils.image_plane_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=img_n, pixel_scale=img_pix, n_channels=len(frequencies)
    )
    kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )
    lens_centre_fixed = (
        settings["lens_mass_model"]["centre_0"],
        settings["lens_mass_model"]["centre_1"],
    )
    kinms_model = _kinms_parametric_model(settings)
    par_inst = kinms_utils.make_instance_from_grid(
        grid_3d=kinms_grid_3d, z_step_kms=z_step_kms, attach_grid=True
    )
    par_analysis, _ = _build_analysis(
        settings, par_inst, lens_centre_fixed, kinms_grid_3d
    )
    ll = _log_likelihood(par_analysis, kinms_model, truth_vector)
    rms = _moment0_dirty_rms(par_analysis, kinms_model, truth_vector)

    print("\n=== Parametric KinMS sanity check (no phase 1) ===\n")
    print(f"  z_step_kms (from MS frequencies) = {z_step_kms:.4f}")
    print(f"  KinMS velocity span = {len(frequencies) * z_step_kms:.1f} km/s")
    print(f"  KinMS source grid = {autolens_utils.source_grid_label_from_settings(settings)}")
    print(f"  Image plane grid = {img_n}² ({img_width}″ field)")
    print(f"  Lens centre (fixed) = {lens_centre_fixed}")
    print(f"  Truth vector = {truth_vector}")
    print(f"  log L (truth vector) = {ll:.2f}")
    print(f"  visibility residual RMS = {rms:.4g}\n")
    print(
        "  A good parametric mock should give a high log L and low residual RMS "
        "at the truth vector after the z_step_kms fix.\n"
    )


def _run_pixelized_comparison(settings, frequencies, z_step_kms, phase1_result):
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
        phase1_result=phase1_result,
    )
    sb_map, lens_centre_phase1 = reconstruction.source_sb_on_grid(
        result=phase1_result,
        grid_2d=source_grid_3d.grid_2d,
    )
    lens_centre_fixed = (
        settings["lens_mass_model"]["centre_0"],
        settings["lens_mass_model"]["centre_1"],
    )

    n_img, pix_scale, width = autolens_utils.source_grid_from_settings(settings)
    image_grid_3d = Grid3D.uniform(
        n_pixels=n_img, pixel_scale=pix_scale, n_channels=len(frequencies)
    )

    kinms_model = _kinms_parametric_model(settings)
    pixel_model = af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(
                profiles.kinMSPixelized, settings["priors"]
            )
        )
    )

    int_flux = kinms_utils.kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=source_grid_3d.n_channels,
        sb_input_units=settings.get("reconstruction", {}).get(
            "sb_input_units", "jy_per_pixel_per_channel"
        ),
    )

    print("\n=== Pipeline differences (pixelized phase 2 vs full parametric) ===\n")
    rows = [
        ("Source grid", autolens_utils.source_grid_label_from_settings(settings, phase1_result), f"{n_img}² ({width}″ field)"),
        ("SB model", "phase-1 source-plane map → inClouds", "exp(−r / re) radial disk"),
        ("Total flux", f"fixed intFlux={int_flux:.4f} Jy km/s", "free intensity prior"),
        ("Lens centre", f"phase-1 fit {lens_centre_phase1}", f"fixed JSON {lens_centre_fixed}"),
        ("Phase-2 params", "7 kinematic", "10 (inc. I, re, centre)"),
        ("KinMS path", "inClouds + flux_clouds", "sbProf + intFlux"),
    ]
    for label, pix, par in rows:
        print(f"  {label:14}  pixelized: {pix}")
        print(f"  {'':14}  parametric: {par}\n")

    print(f"  Phase-1 lens − fixed lens: Δcentre = "
          f"({lens_centre_phase1[0]-lens_centre_fixed[0]:+.4f}, "
          f"{lens_centre_phase1[1]-lens_centre_fixed[1]:+.4f}) arcsec\n")

    # --- Pixelized reference (should be good) ---
    pix_inst = kinms_utils.make_pixelized_instance_from_grid(
        grid_3d=source_grid_3d,
        z_step_kms=z_step_kms,
        sb_map=sb_map,
        sb_input_units=settings.get("reconstruction", {}).get(
            "sb_input_units", "jy_per_pixel_per_channel"
        ),
        **kinms_utils.pixelized_instance_kwargs_from_settings(settings),
    )
    pix_analysis, _ = _build_analysis(
        settings, pix_inst, lens_centre_phase1, source_grid_3d
    )
    truth_pix = _truth_vector_for_model(
        settings=settings,
        model=pixel_model,
        fallback_vector=TRUTH_KINMS_VECTOR,
    )
    ll_pix = _log_likelihood(pix_analysis, pixel_model, truth_pix)
    rms_pix = _moment0_dirty_rms(pix_analysis, pixel_model, truth_pix)

    # --- Parametric on coarse image grid (current kinms_mock_parametric.json) ---
    par_inst_40 = kinms_utils.make_instance_from_grid(
        grid_3d=image_grid_3d, z_step_kms=z_step_kms, attach_grid=True
    )
    for centre, label in [
        (lens_centre_fixed, "fixed JSON lens centre"),
        (lens_centre_phase1, "phase-1 lens centre"),
    ]:
        par_analysis, _ = _build_analysis(settings, par_inst_40, centre, image_grid_3d)
        truth_par = _truth_vector_for_model(
            settings=settings,
            model=kinms_model,
            fallback_vector=TRUTH_KINMS_VECTOR,
        )
        ll = _log_likelihood(par_analysis, kinms_model, truth_par)
        rms = _moment0_dirty_rms(par_analysis, kinms_model, truth_par)
        print(f"Parametric 40² + {label}:")
        print(f"  log L (truth vector) = {ll:.2f}")
        print(f"  visibility residual RMS = {rms:.4g}\n")

    # --- Parametric on same 256² source grid as pixelized ---
    par_inst_256 = kinms_utils.make_instance_from_grid(
        grid_3d=source_grid_3d, z_step_kms=z_step_kms, attach_grid=True
    )
    par_analysis_hi, _ = _build_analysis(
        settings, par_inst_256, lens_centre_phase1, source_grid_3d
    )
    truth_par = _truth_vector_for_model(
        settings=settings,
        model=kinms_model,
        fallback_vector=TRUTH_KINMS_VECTOR,
    )
    ll_par_hi = _log_likelihood(par_analysis_hi, kinms_model, truth_par)
    rms_par_hi = _moment0_dirty_rms(par_analysis_hi, kinms_model, truth_par)

    print("Pixelized 256² + phase-1 lens (reference):")
    print(f"  log L (truth kinematics) = {ll_pix:.2f}")
    print(f"  visibility residual RMS = {rms_pix:.4g}\n")

    print("Parametric 256² + phase-1 lens (matched grid, still exp disk):")
    print(f"  log L (truth vector) = {ll_par_hi:.2f}")
    print(f"  visibility residual RMS = {rms_par_hi:.4g}\n")

    print("Interpretation:")
    print("  • If pixelized ≫ parametric(40²) but parametric(256²) improves, grid resolution")
    print("    and/or lens-centre mismatch are the main issues.")
    print("  • If parametric(256²) still ≪ pixelized, the exponential disk SB is too rigid")
    print("    (phase-1 map is doing most of the spatial work in pixelized mode).")
    print("  • Try parametric_flux_from_phase1 next to isolate intensity from SB shape.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose parametric vs pixelized KinMS phase-2 differences."
    )
    parser.add_argument(
        "--settings",
        default="settings/runners/kinms_mock_pixelized.json",
        help="Settings JSON. Parametric JSON runs a phase-1-free sanity check.",
    )
    parser.add_argument(
        "--run-phase1",
        action="store_true",
        help="Run phase-1 reconstruction (required for pixelized comparison).",
    )
    parser.add_argument(
        "--phase1-result",
        default=None,
        help="Path to an existing phase-1 search output folder (optional).",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    frequencies, _, _, _ = load_cube_data(settings)
    z_step_kms = _spectral_setup(settings, frequencies)

    if "reconstruction" not in settings:
        _run_parametric_sanity_check(settings, frequencies, z_step_kms)
        return

    if args.phase1_result:
        raise NotImplementedError(
            "Loading a saved phase-1 result is not implemented yet; use --run-phase1."
        )
    if not args.run_phase1:
        parser.error(
            "Pixelized settings require phase 1. Pass --run-phase1, or use "
            "settings/runners/kinms_mock_parametric.json for a parametric-only check."
        )

    phase1_result = reconstruction.run_reconstruction(settings)
    _run_pixelized_comparison(settings, frequencies, z_step_kms, phase1_result)


if __name__ == "__main__":
    main()
