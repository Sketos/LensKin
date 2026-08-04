#!/usr/bin/env python3
"""
Smoke-test GalPaK modes 1 and 2 on the unlensed KinMS mock and write fit plots.

Mode 1: truth-parameter GalPaK forward model vs data.
Mode 2: short phase-1 reconstruction, then GalPaK with intensity fixed from
phase-1 flux and remaining parameters at truth.

Writes under ``output/galpak_mock_unlensed_smoke/plots/``.

Note: the mock cube was generated with KinMS (arctan rotation), while GalPaK
uses an isothermal curve here — expect imperfect residuals; these plots are
a wiring / scale check, not a self-consistency proof.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

# GalPaK 1.34 imports pkg_resources; newer setuptools may omit it.
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        _pr = types.ModuleType("pkg_resources")

        class _Dist:
            version = "1.34.0"

        def _get_distribution(_name):
            return _Dist()

        _pr.get_distribution = _get_distribution
        _pr.DistributionNotFound = Exception
        sys.modules["pkg_resources"] = _pr

for _parent in Path(__file__).resolve().parents:
    if (_parent / "scripts" / "bootstrap.py").is_file():
        sys.path.insert(0, str(_parent))
        break

from scripts.bootstrap import setup

setup(__file__)

import matplotlib.pyplot as plt
import numpy as np

from scripts.smoke_unlensed_three_modes import (
    _analysis_for_settings,
    _save_kinematic_fit_plots,
)
from scripts.test_truth_model import _visibility_fit_stats
from src.analysis import analysis as analysis_mod
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.pipelines import reconstruction
from src.pipelines.lens_model import validate_lensing_settings
from src.pipelines.normalization import (
    PARAMETRIC,
    PARAMETRIC_FLUX_FROM_PHASE1,
    intensity_from_phase1_intflux,
    validate_normalization_settings,
)
from src.pipelines.pixelized_plots import plot_phase1_fit
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/galpak_selfconsist_smoke"
PLOTS = OUT / "plots"


def _smoke_mode1(settings):
    validate_lensing_settings(settings)
    assert validate_normalization_settings(settings) == PARAMETRIC
    analysis = _analysis_for_settings(
        settings,
        instance_obj=None,
        tracer=build_tracer(settings),
    )
    truth = truth_instance_from_settings(settings)
    model_data = analysis.model_data_from_instance(instance=truth)
    stats = _visibility_fit_stats(analysis, model_data)
    plots = _save_kinematic_fit_plots(
        analysis,
        model_data,
        PLOTS / "parametric",
        label="GalPaK mode1",
        settings=settings,
    )
    return {
        "mode": "parametric",
        "ok": True,
        "chi_squared_per_datum": stats["chi_squared_per_datum"],
        "log_likelihood": stats["log_likelihood"],
        "plots": plots,
    }


def _smoke_mode2(settings):
    validate_lensing_settings(settings)
    assert validate_normalization_settings(settings) == PARAMETRIC_FLUX_FROM_PHASE1

    settings = copy.deepcopy(settings)
    settings["output_path"] = str(OUT / "phase1_run")
    settings["plot"] = False
    settings["reconstruction"]["search"]["maxiter"] = min(
        int(settings["reconstruction"]["search"].get("maxiter", 200)),
        40,
    )
    settings["reconstruction"]["search"]["number_of_cores"] = 1

    result = reconstruction.run_reconstruction(settings)
    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)
    n_pixels, pixel_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_plane_grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=len(frequencies),
    )
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
        phase1_result=result,
    )
    flux_snr_threshold = reconstruction.flux_snr_threshold_from_settings(settings)
    sb_map, lens_centre, sb_full, _noise = reconstruction.source_sb_for_phase2_flux(
        result=result,
        grid_2d=source_grid_3d.grid_2d,
        snr_threshold=flux_snr_threshold,
    )
    sb_input_units = settings["reconstruction"].get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    intflux = kinms_utils.kinms_intflux_from_sb_map(
        sb_map=sb_map,
        z_step_kms=z_step_kms,
        n_channels=source_grid_3d.n_channels,
        sb_input_units=sb_input_units,
    )
    galpak_intensity = intensity_from_phase1_intflux(
        "GalPak", intflux, z_step_kms
    )
    dataset_instance = type(
        "GalPaKInstance",
        (),
        {"grid_3d": source_grid_3d, "int_flux": galpak_intensity},
    )()

    mask_3d = Mask3D.unmasked(
        n_channels=image_plane_grid_3d.n_channels,
        shape_2d=image_plane_grid_3d.shape_2d,
        pixel_scales=image_plane_grid_3d.pixel_scales,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
        settings=settings,
    )
    analysis = analysis_mod.Analysis(
        masked_dataset=MaskedDataset(
            dataset=Dataset(
                uv_wavelengths=uv_wavelengths,
                visibilities=visibilities,
                noise_map=sigma,
                z_step_kms=z_step_kms,
            ),
            mask_3d=mask_3d,
            instance=dataset_instance,
        ),
        transformers=transformers,
        tracer=build_tracer(settings, centre=lens_centre),
        settings=settings,
    )
    truth = truth_instance_from_settings(settings)
    truth.galaxies.source.intensity = float(galpak_intensity)
    model_data = analysis.model_data_from_instance(instance=truth)
    stats = _visibility_fit_stats(analysis, model_data)

    phase1_dir = PLOTS / "parametric_flux" / "phase1"
    plot_phase1_fit(
        result=result,
        sb_map=sb_full,
        output_dir=phase1_dir,
        settings=settings,
        sb_map_extent=autolens_utils.image_extent_from_bounding_box(
            reconstruction.source_plane_bounding_box_from_result(result)
        ),
    )
    kin_plots = _save_kinematic_fit_plots(
        analysis,
        model_data,
        PLOTS / "parametric_flux" / "phase2",
        label="GalPaK mode2",
        settings=settings,
    )
    return {
        "mode": "parametric_flux_from_phase1",
        "ok": True,
        "flux_snr_threshold": flux_snr_threshold,
        "phase1_intflux_jy_kms": float(intflux),
        "galpak_intensity": float(galpak_intensity),
        "chi_squared_per_datum": stats["chi_squared_per_datum"],
        "log_likelihood": stats["log_likelihood"],
        "plots": {
            "phase1": str(phase1_dir / "fit_triplet.png"),
            "phase1_sb_map": str(phase1_dir / "sb_map.png"),
            **{f"phase2_{k}": v for k, v in kin_plots.items()},
        },
    }


def _write_compare_figure(rows):
    """
    Stack the existing dirty_mom0_fit PNGs into one figure.

    Uses a pixel collage (no ``imshow`` re-render) so colour scales match the
    individual ``dirty_mom0_fit.png`` files exactly.
    """
    from PIL import Image, ImageDraw, ImageFont

    panels = []
    p1 = PLOTS / "parametric" / "dirty_mom0_fit.png"
    p2 = PLOTS / "parametric_flux" / "phase2" / "dirty_mom0_fit.png"
    if p1.is_file() and rows:
        panels.append((p1, rows[0], "mode 1 parametric"))
    if p2.is_file() and len(rows) > 1:
        panels.append((p2, rows[1], "mode 2 flux-from-phase1"))
    if not panels:
        return None

    images = [Image.open(path).convert("RGB") for path, _, _ in panels]
    width = max(im.width for im in images)
    header_h = 36
    gap = 12
    total_h = sum(im.height for im in images) + header_h * len(images) + gap * (
        len(images) - 1
    )
    canvas = Image.new("RGB", (width, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    y = 0
    for im, (path, row, label) in zip(images, panels):
        title = (
            f"GalPaK {label}  χ²/N={row['chi_squared_per_datum']:.3g}  "
            f"lnL={row['log_likelihood']:.3g}"
        )
        draw.rectangle([0, y, width, y + header_h], fill=(255, 255, 255))
        draw.text((8, y + 8), title, fill=(0, 0, 0), font=font)
        y += header_h
        x0 = (width - im.width) // 2
        canvas.paste(im, (x0, y))
        y += im.height + gap

    out = PLOTS / "mom0_compare.png"
    canvas.save(out)
    return str(out)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=== GalPaK mode 1 (parametric) ===", flush=True)
    s1 = load_settings(str(ROOT / "settings/runners/galpak_mock_unlensed_parametric.json"))
    rows.append(_smoke_mode1(s1))
    print(json.dumps(rows[-1], indent=2), flush=True)

    print("\n=== GalPaK mode 2 (flux from phase-1) ===", flush=True)
    s2 = load_settings(
        str(ROOT / "settings/runners/galpak_mock_unlensed_parametric_flux.json")
    )
    rows.append(_smoke_mode2(s2))
    print(json.dumps(rows[-1], indent=2), flush=True)

    compare = _write_compare_figure(rows)
    summary = {"rows": rows, "mom0_compare": compare}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nDONE", OUT, flush=True)
    if compare:
        print("compare:", compare, flush=True)


if __name__ == "__main__":
    main()
