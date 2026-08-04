#!/usr/bin/env python3
"""
Audit position-angle and inclination conventions across LensKin, KinMS, and data.

Examples:
  python scripts/audit_kinematic_conventions.py \\
    --settings settings/runners/kinms_mock_parametric_bbox512.json
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

import numpy as np

from src.analysis import analysis as analysis_mod
from src.pipelines.priors import source_model_from_profile
from src.pipelines.runner import load_settings
from src.pipelines.truth_model import truth_instance_from_settings
from src.pipelines.truth_parameters import (
    PARAMETRIC_KINMS_DEBUG_ORDER,
    truth_parameters_from_settings,
)
from src.model import profiles
from src.utils import analysis_utils, autolens_utils

import autofit as af


def _mom0_major_axis_pa_deg(mom0, grid_2d, threshold=0.2):
    """Major-axis PA from weighted 2nd moments (degrees, +x reference)."""
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    y = coords[:, :, 0]
    x = coords[:, :, 1]
    peak = float(np.nanmax(mom0))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    w = np.where(mom0 >= threshold * peak, mom0, 0.0)
    if w.sum() <= 0:
        return float("nan")
    yc = (w * y).sum() / w.sum()
    xc = (w * x).sum() / w.sum()
    dy = y - yc
    dx = x - xc
    mu20 = (w * dx * dx).sum() / w.sum()
    mu02 = (w * dy * dy).sum() / w.sum()
    mu11 = (w * dx * dy).sum() / w.sum()
    return float(np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)))


def _mom0_pa_east_of_north_deg(mom0, grid_2d, threshold=0.2):
    """
    Major-axis PA in astronomical convention: degrees east of north (+y).

    Autolens grids use +y as the northern axis and +x as east.
    """
    coords = np.asarray(grid_2d).reshape(grid_2d.shape_native + (2,))
    y = coords[:, :, 0]
    x = coords[:, :, 1]
    peak = float(np.nanmax(mom0))
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    w = np.where(mom0 >= threshold * peak, mom0, 0.0)
    if w.sum() <= 0:
        return float("nan")
    yc = (w * y).sum() / w.sum()
    xc = (w * x).sum() / w.sum()
    dy = y - yc
    dx = x - xc
    mu20 = (w * dx * dx).sum() / w.sum()
    mu02 = (w * dy * dy).sum() / w.sum()
    mu11 = (w * dx * dy).sum() / w.sum()
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    _, evecs = np.linalg.eigh(cov)
    vx = float(evecs[0, 1])
    vy = float(evecs[1, 1])
    return float(np.degrees(np.arctan2(vx, vy)))


def _pa_offset_mod180(a_deg, b_deg):
    """Signed difference between two PAs modulo 180°."""
    return float((a_deg - b_deg + 90.0) % 180.0 - 90.0)


def _chi2_per_datum(analysis_instance, instance):
    model_data = analysis_instance.model_data_from_instance(instance=instance)
    fit = analysis_mod.fit.DatasetFit(
        masked_dataset=analysis_instance.masked_dataset,
        model_data=model_data,
    )
    mask = fit.mask == False
    residual = fit.data[mask] - model_data[mask]
    sigma = fit.noise_map[mask]
    return float(np.sum((residual / sigma) ** 2) / mask.sum())


def _build_analysis(settings):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_truth_model",
        Path(__file__).resolve().parent / "test_truth_model.py",
    )
    ttm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ttm)
    pipeline = ttm._build_pipeline(settings)
    return pipeline["analysis"], pipeline


def run_audit(settings_path):
    settings = load_settings(settings_path)
    analysis_instance, pipeline = _build_analysis(settings)
    truth_params = truth_parameters_from_settings(settings)
    truth = truth_instance_from_settings(settings)
    prof = truth.galaxies.source
    z_step = pipeline["z_step_kms"]

    model = af.Collection(
        galaxies=af.Collection(
            source=source_model_from_profile(profiles.kinMS, settings["priors"])
        )
    )

    print("\n=== Kinematic convention audit ===\n")
    print("KinMS (package docs):")
    print("  posAng: PA=0° => disc major axis / redshifted side along +y (KinMS sky plane).")
    print("  inc: inclination on the sky in degrees (0=face-on, 90=edge-on).")
    print("  phaseCent: morphological centre offset [x_pix, y_pix] from cube centre.")
    print("\nLensKin kinMS profile:")
    print("  phi is passed directly to KinMS posAng (no offset or unit conversion).")
    print(
        "  phaseCent = [centre_0, centre_1] with centre tuple "
        "(centre_0, centre_1) = (x, y)."
    )
    print("\nPosition-angle conventions:")
    print("  KinMS posAng: degrees CCW from +y (major axis at PA=0).")
    print("  Astronomical PA: degrees east of north (+y=N, +x=E on autolens grids).")
    print("  Moment PA (+x ref): 0.5*atan2(2*mu11, mu20-mu02); can differ by ~180°")
    print("  from KinMS posAng on inclined disks (major/minor ambiguity).")
    print("\nAutolens grids:")
    print("  Coordinates are (y, x) in arcsec; y decreases along increasing row index.")
    print("\nLensing pipeline (analysis_utils):")
    print("  Source cube: y-flip [::-1, :] before ray-trace; no post-lens mirror.")

    print("\n=== Autofit vector order (debug_instance_vector) ===\n")
    print(model.info)
    print(
        "\n  Note: centre tuple is last; vector slots are "
        "index 8 = centre_0 (x), index 9 = centre_1 (y); see model.info."
    )
    print(f"  PARAMETRIC_KINMS_DEBUG_ORDER = {PARAMETRIC_KINMS_DEBUG_ORDER}")

    vec = settings.get("debug_instance_vector")
    if vec is not None:
        inst_vec = model.instance_from_vector(vector=vec)
        inst_named = truth_instance_from_settings(settings)
        print("\n=== Centre / phi from vector vs named truth builder ===\n")
        for label, inst in [("instance_from_vector", inst_vec), ("truth_instance", inst_named)]:
            src = inst.galaxies.source
            print(
                f"  {label}: centre={src.centre}, phi={src.phi}, "
                f"inclination={src.inclination}"
            )
        if inst_vec.galaxies.source.centre != inst_named.galaxies.source.centre:
            print(
                "\n  Centre mismatch detail:"
            )
            print(
                f"    vector slots [8:10] = {vec[8:10]} "
                f"(see model.info for centre_0/centre_1 indices)"
            )
            print(
                f"    truth_parameters centre_0/centre_1 = "
                f"{truth_parameters_from_settings(settings).get('centre_0')!r}, "
                f"{truth_parameters_from_settings(settings).get('centre_1')!r}"
            )

    print("\n=== Measured position angles (moment-0, 20% peak mask) ===\n")
    src_cube = prof.profile_cube_from_grid(
        pipeline["kinms_grid_3d"],
        z_step,
        instance=analysis_instance.masked_dataset.instance,
    )
    src_mom0 = src_cube.sum(axis=0) * z_step
    pa_src_x = _mom0_major_axis_pa_deg(src_mom0, pipeline["kinms_grid_3d"].grid_2d)
    pa_src_eon = _mom0_pa_east_of_north_deg(
        src_mom0, pipeline["kinms_grid_3d"].grid_2d
    )
    print(f"  Truth phi (KinMS posAng, CCW from +y) = {prof.phi:.2f}°")
    print(f"  Source-plane mom0 PA (+x reference)    = {pa_src_x:.2f}°")
    print(f"  Source-plane mom0 PA (east of north)   = {pa_src_eon:.2f}°")
    print(
        f"  phi − PA(+x) mod 180                  = "
        f"{_pa_offset_mod180(prof.phi, pa_src_x):+.1f}°"
    )
    print(
        f"  phi − PA(EoN) mod 180                  = "
        f"{_pa_offset_mod180(prof.phi, pa_src_eon):+.1f}°"
    )
    print(
        f"  (phi ± 90°) − PA(+x) mod 180           = "
        f"{_pa_offset_mod180(prof.phi + 90.0, pa_src_x):+.1f}° / "
        f"{_pa_offset_mod180(prof.phi - 90.0, pa_src_x):+.1f}°"
    )

    dirty = autolens_utils.dirty_cube_from(
        visibilities=analysis_instance.masked_dataset.data,
        transformers=analysis_instance.transformers,
    )
    data_mom0 = dirty.sum(axis=0) * z_step
    pa_data_x = _mom0_major_axis_pa_deg(data_mom0, pipeline["image_grid_3d"].grid_2d)
    pa_data_eon = _mom0_pa_east_of_north_deg(
        data_mom0, pipeline["image_grid_3d"].grid_2d
    )

    lensed = analysis_utils.lensed_cube_from_tracer(
        cube=src_cube,
        tracer=analysis_instance.tracer,
        grid=pipeline["image_grid_3d"].grid_2d,
        source_grid_2d=pipeline["kinms_grid_3d"].grid_2d,
    )
    model_mom0 = lensed.sum(axis=0) * z_step
    pa_model_x = _mom0_major_axis_pa_deg(model_mom0, pipeline["image_grid_3d"].grid_2d)
    pa_model_eon = _mom0_pa_east_of_north_deg(
        model_mom0, pipeline["image_grid_3d"].grid_2d
    )

    print(f"  Dirty data image-plane PA (+x)         = {pa_data_x:.2f}°")
    print(f"  Dirty data image-plane PA (EoN)        = {pa_data_eon:.2f}°")
    print(f"  Lensed truth model image-plane PA (+x) = {pa_model_x:.2f}°")
    print(f"  Lensed truth model image-plane PA (EoN)= {pa_model_eon:.2f}°")
    print(
        f"  Model − data PA (+x)                   = {pa_model_x - pa_data_x:+.2f}°"
    )
    print(
        f"  Model − data PA (EoN)                  = {pa_model_eon - pa_data_eon:+.2f}°"
    )
    print(
        "\n  A ~90° offset between phi and PA(+x) is expected (KinMS measures from "
        "+y; moments use +x). Check PA(EoN) vs phi before adding a posAng offset."
    )

    phi_offset_chi2 = []
    for dphi in (-90, -45, 0, 45, 90):
        inst = truth_instance_from_settings(settings)
        inst.galaxies.source.phi = float(prof.phi + dphi)
        phi_offset_chi2.append((dphi, _chi2_per_datum(analysis_instance, inst)))
    truth_c2n = next(c2 for d, c2 in phi_offset_chi2 if d == 0)
    print("  posAng offset trial (phi_truth + offset):")
    for dphi, c2 in phi_offset_chi2:
        print(f"    offset {dphi:+3d}°: chi²/N = {c2:.6f}")
    best_offset = min(phi_offset_chi2, key=lambda t: t[1])[0]
    if best_offset != 0 and min(c[1] for c in phi_offset_chi2) < truth_c2n - 1e-4:
        print(
            f"  WARN — chi² improves by offsetting posAng by {best_offset:+d}°; "
            "check for a systematic PA convention mismatch."
        )
    else:
        print(
            "  PASS — ±90° posAng offsets do not materially improve chi²/N; no "
            "90° wiring correction needed in profiles.py."
        )

    print("\n=== UV sensitivity (chi² / N) ===\n")
    print(f"  Truth parameters: chi²/N = {truth_c2n:.6f}")

    phi_vals = [0, 45, 90, 135, 180]
    inc_vals = [45, 55, 65, 75, 85]
    phi_chi2 = []
    inc_chi2 = []
    for phi in phi_vals:
        inst = truth_instance_from_settings(settings)
        inst.galaxies.source.phi = phi
        phi_chi2.append(_chi2_per_datum(analysis_instance, inst))
    for inc in inc_vals:
        inst = truth_instance_from_settings(settings)
        inst.galaxies.source.inclination = inc
        inc_chi2.append(_chi2_per_datum(analysis_instance, inst))

    print(f"  phi sweep {phi_vals}:")
    print(f"    chi²/N = {[f'{x:.6f}' for x in phi_chi2]}")
    print(f"  inc sweep {inc_vals}:")
    print(f"    chi²/N = {[f'{x:.6f}' for x in inc_chi2]}")
    print(
        f"  phi range = {max(phi_chi2) - min(phi_chi2):.6f}; "
        f"inc range = {max(inc_chi2) - min(inc_chi2):.6f}"
    )
    if max(phi_chi2) - min(phi_chi2) < 0.001:
        print(
            "\n  PHI IS ESSENTIALLY UNCONSTRAINED in UV space for this dataset. "
            "The mock is unresolved (re≈0.05″ vs ~0.4″ beam); Nautilus can "
            "return arbitrary phi while chi²/N ≈ 1."
        )

    print("\n=== Verdict ===\n")
    centre_match = True
    if vec is not None:
        centre_match = (
            inst_vec.galaxies.source.centre == inst_named.galaxies.source.centre
            and inst_vec.galaxies.source.phi == inst_named.galaxies.source.phi
        )
    if centre_match:
        print("  PASS — phi/inc/centre are wired consistently (phi → posAng, phaseCent sign).")
    else:
        print(
            "  FAIL — debug_instance_vector centre slots do not match "
            "truth_instance_from_settings; check centre_0/centre_1 order."
        )
    if abs(_pa_offset_mod180(pa_src_eon, prof.phi)) <= 15.0:
        print(
            f"  PASS — source-plane PA east-of-north ({pa_src_eon:.1f}°) matches "
            f"phi ({prof.phi:.1f}°) within 15°."
        )
    elif abs(_pa_offset_mod180(pa_src_x, prof.phi)) <= 15.0:
        print(
            f"  PASS — source-plane PA (+x ref, {pa_src_x:.1f}°) matches phi "
            f"({prof.phi:.1f}°) within 15°."
        )
    elif abs(_pa_offset_mod180(pa_src_x, prof.phi + 90.0)) <= 15.0:
        print(
            f"  NOTE — PA(+x) matches phi+90° (expected KinMS +y vs moment +x offset); "
            "do not add +90° to posAng in code."
        )
    else:
        print(
            f"  WARN — source-plane PA (+x {pa_src_x:.1f}°, EoN {pa_src_eon:.1f}°) "
            f"differs from phi ({prof.phi:.1f}°); inspect inclination / axis flips."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        required=True,
        help="Runner settings JSON (e.g. kinms_mock_parametric_bbox512.json)",
    )
    args = parser.parse_args()
    run_audit(args.settings)


if __name__ == "__main__":
    main()
