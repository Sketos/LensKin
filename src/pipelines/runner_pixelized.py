import argparse
import copy
import json
from pathlib import Path

import autofit as af
import numpy as np
from astropy import units
from astropy.io import fits

from src.analysis import analysis
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.model import profiles
from src.pipelines import reconstruction
from src.pipelines.normalization import (
    PARAMETRIC_FLUX_FROM_PHASE1,
    PIXELIZED,
    intensity_from_phase1_intflux,
    priors_for_normalization_mode,
    validate_normalization_settings,
)
from src.pipelines.pixelized_plots import (
    plot_dirty_data,
    plot_phase1_fit,
    plot_phase2_fit,
    plots_root_from_settings,
)
from src.pipelines.lens_model import (
    free_lens_centre_from_settings,
    lens_galaxy_model_from_settings,
    validate_lensing_settings,
)
from src.pipelines.runner import build_tracer, load_cube_data, load_settings
from src.pipelines.priors import source_model_from_profile
from src.pipelines.search import build_search_from_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils


def _phase2_model_from_settings(settings, profile, priors_cfg, *, phase1_lens_centre=None):
    """
    Phase-2 autofit model: kinematic source, optionally free lens centre.

    When ``free_lens_centre`` is enabled, lens-centre priors are centred on the
    phase-1 fitted centre (falling back to ``lens_mass_model``).
    """
    source = source_model_from_profile(profile, priors_cfg)
    if not free_lens_centre_from_settings(settings):
        return af.Collection(galaxies=af.Collection(source=source))

    model_settings = settings
    if phase1_lens_centre is not None:
        model_settings = copy.deepcopy(settings)
        model_settings["lens_mass_model"] = dict(settings["lens_mass_model"])
        model_settings["lens_mass_model"]["centre_0"] = float(phase1_lens_centre[0])
        model_settings["lens_mass_model"]["centre_1"] = float(phase1_lens_centre[1])

    return af.Collection(
        galaxies=af.Collection(
            lens=lens_galaxy_model_from_settings(model_settings, free_centre=True),
            source=source,
        )
    )


def run_from_settings(settings):
    validate_lensing_settings(settings)
    mode = validate_normalization_settings(settings)

    af.conf.instance.push(
        new_path=settings.get("config_path", "./config"),
        output_path=settings["output_path"],
    )

    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    autolens_utils.resolve_image_plane_grid_in_settings(settings, uv_wavelengths)

    n_pixels, pixel_scale, _ = autolens_utils.image_plane_grid_from_settings(settings)
    image_plane_grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=len(frequencies),
    )

    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)

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

    plot_enabled = settings.get("plot", True)
    plots_root = plots_root_from_settings(settings)
    if plot_enabled:
        _, _, real_space_width = autolens_utils.source_grid_from_settings(settings)
        kin_extent = autolens_utils.image_extent_arcsec(real_space_width)
        plot_dirty_data(
            visibilities=visibilities,
            transformers=transformers,
            output_dir=plots_root / "data",
            extent=kin_extent,
        )

    # Phase 1: pixelized source reconstruction on velocity-averaged data
    reconstruction_result = reconstruction.run_reconstruction(settings)
    source_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings,
        n_channels=len(frequencies),
        phase1_result=reconstruction_result,
    )
    flux_snr_threshold = reconstruction.flux_snr_threshold_from_settings(settings)
    sb_map, lens_centre, sb_full, _noise_map = (
        reconstruction.source_sb_for_phase2_flux(
            result=reconstruction_result,
            grid_2d=source_grid_3d.grid_2d,
            snr_threshold=flux_snr_threshold,
        )
    )
    if flux_snr_threshold is not None:
        print(
            f"  Phase-1 flux SNR cut: threshold={flux_snr_threshold:g} "
            f"(kept {(np.asarray(sb_map) != 0).sum()} / {np.asarray(sb_map).size} px)"
        )

    if plot_enabled:
        plot_phase1_fit(
            result=reconstruction_result,
            sb_map=sb_full,
            output_dir=plots_root / "phase1",
            settings=settings,
            sb_map_extent=autolens_utils.image_extent_from_bounding_box(
                reconstruction.source_plane_bounding_box_from_result(
                    reconstruction_result
                )
            ),
        )

    flux_threshold = settings.get("reconstruction", {}).get("flux_threshold", 0.0)
    sb_input_units = settings.get("reconstruction", {}).get(
        "sb_input_units", "jy_per_pixel_per_channel"
    )
    cloud_kwargs = kinms_utils.pixelized_instance_kwargs_from_settings(settings)

    if mode == PIXELIZED:
        profile = profiles.kinMSPixelized
        dataset_instance = kinms_utils.make_pixelized_instance_from_grid(
            grid_3d=source_grid_3d,
            z_step_kms=z_step_kms,
            sb_map=sb_map,
            flux_threshold=flux_threshold,
            sb_input_units=sb_input_units,
            **cloud_kwargs,
        )
        total_flux = None
    elif mode == PARAMETRIC_FLUX_FROM_PHASE1:
        model_name = settings["model_name"]
        # Phase-1 map → velocity-integrated flux (Jy km/s), after optional SNR cut.
        intflux_jy_kms = kinms_utils.kinms_intflux_from_sb_map(
            sb_map=sb_map,
            z_step_kms=z_step_kms,
            n_channels=source_grid_3d.n_channels,
            sb_input_units=sb_input_units,
        )
        total_flux = intensity_from_phase1_intflux(
            model_name=model_name,
            intflux_jy_kms=intflux_jy_kms,
            z_step_kms=z_step_kms,
        )
        if model_name == "GalPak":
            profile = profiles.GalPaK
            # GalPaK builds on this source-plane grid; Analysis regrids to the
            # image plane (same path as KinMS when instance.grid_3d is set).
            dataset_instance = type(
                "GalPaKInstance",
                (),
                {"grid_3d": source_grid_3d, "int_flux": total_flux},
            )()
            print(
                f"  Phase-2 GalPaK intensity fixed from phase-1 flux: "
                f"{total_flux:.6g} (cube sum; intFlux={intflux_jy_kms:.6g} Jy km/s)"
            )
        elif model_name == "KinMS":
            profile = profiles.kinMS
            dataset_instance = kinms_utils.make_instance_from_grid(
                grid_3d=source_grid_3d,
                z_step_kms=z_step_kms,
                attach_grid=True,
                disk_thick=cloud_kwargs["scale_height_arcsec"],
            )
            dataset_instance.int_flux = total_flux
            print(
                f"  Phase-2 KinMS intensity fixed from phase-1 flux: "
                f"{total_flux:.6g} Jy km/s"
            )
        else:
            raise ValueError(
                f"Unsupported model_name for parametric_flux_from_phase1: "
                f"{model_name!r}"
            )
    else:
        raise ValueError(
            f"Unsupported normalization_mode for two-phase runner: {mode}"
        )

    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
    )
    free_lens_centre = free_lens_centre_from_settings(settings)
    # Fixed tracer from phase 1 unless phase 2 re-optimizes lens centre.
    tracer = None if free_lens_centre else build_tracer(settings, centre=lens_centre)
    if free_lens_centre:
        print(
            f"  Phase 2: free lens centre (prior centred on phase-1 "
            f"({lens_centre[0]:.4f}, {lens_centre[1]:.4f}), "
            f"half-width={settings.get('lens_centre_half_width', 0.05)}\")"
        )
    else:
        print(
            f"  Phase 2: lens centre fixed at "
            f"({lens_centre[0]:.4f}, {lens_centre[1]:.4f})"
        )

    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
        instance=dataset_instance,
    )
    analysis_instance = analysis.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
        settings=settings,
    )

    if settings.get("write_dirty_data_cube", True):
        fits.writeto(
            settings.get("dirty_data_cube_filename", "./dirty_data_cube.fits"),
            data=autolens_utils.dirty_cube_from(
                visibilities=dataset.visibilities,
                transformers=transformers,
            ),
            overwrite=True,
        )

    priors_cfg = priors_for_normalization_mode(
        priors_cfg=settings["priors"],
        mode=mode,
        total_flux=total_flux,
        settings=settings,
    )
    model = _phase2_model_from_settings(
        settings,
        profile,
        priors_cfg,
        phase1_lens_centre=lens_centre,
    )

    search = build_search_from_settings(settings["search"])
    result = search.fit(model=model, analysis=analysis_instance)

    if plot_enabled:
        plot_phase2_fit(
            analysis_instance=analysis_instance,
            result=result,
            output_dir=plots_root / "phase2",
            settings=settings,
        )

    if settings.get("write_dirty_model_cube", True):
        if "debug_instance_vector" in settings:
            fit_instance = model.instance_from_vector(
                vector=settings["debug_instance_vector"]
            )
        else:
            fit_instance = result.max_log_likelihood_instance
        fits.writeto(
            settings.get("dirty_model_cube_filename", "./dirty_model_cube.fits"),
            data=autolens_utils.dirty_cube_from(
                visibilities=analysis_instance.model_data_from_instance(
                    instance=fit_instance
                ),
                transformers=transformers,
            ),
            overwrite=True,
        )

    return result


def main(default_settings_path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        default=str(default_settings_path),
        help="Path to JSON settings file",
    )
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run_from_settings(settings)
