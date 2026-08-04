import argparse
import json
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np
from astropy import units
from astropy.io import fits

from src.analysis import analysis
from src.dataset.dataset import Dataset, MaskedDataset
from src.grid.grid import Grid3D
from src.mask.mask import Mask3D
from src.model import profiles
from src.pipelines.cube_io import load_exported_array
from src.pipelines.normalization import (
    normalization_mode_from_settings,
    priors_for_normalization_mode,
    requires_phase1,
    validate_normalization_settings,
)
from src.pipelines.lens_model import (
    free_lens_centre_from_settings,
    lens_centre_from_instance,
    lens_galaxy_model_from_settings,
    mass_model_for_settings,
    validate_lensing_settings,
)
from src.pipelines.priors import source_model_from_profile
from src.pipelines.search import build_search_from_settings
from src.utils import autolens_utils, kinms_utils, spectral_utils


def load_settings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cube_data(settings):
    directory = Path(settings["data_directory"])
    uids = settings["uids"]
    width = settings["width"]
    patterns = settings["data_patterns"]

    list_of_frequencies = []
    list_of_uv_wavelengths = []
    list_of_visibilities = []
    list_of_sigma = []

    extra_context = settings.get("pattern_context", {})
    for uid in uids:
        fmt = {"uid": uid, "width": width, **extra_context}
        frequencies = load_exported_array(
            directory / patterns["frequencies"].format(**fmt)
        )
        list_of_frequencies.append(frequencies)

        uv_wavelengths = load_exported_array(
            directory / patterns["uv_wavelengths"].format(**fmt)
        )
        list_of_uv_wavelengths.append(np.concatenate((uv_wavelengths, uv_wavelengths), axis=1))

        visibilities = load_exported_array(
            directory / patterns["visibilities"].format(**fmt)
        )
        list_of_visibilities.append(np.concatenate((visibilities[0], visibilities[1]), axis=1))

        sigma = load_exported_array(directory / patterns["sigma"].format(**fmt))
        list_of_sigma.append(np.concatenate((sigma[0], sigma[1]), axis=1))

    if len(uids) == 1:
        frequencies = list_of_frequencies[0]
        uv_wavelengths = np.concatenate(list_of_uv_wavelengths, axis=0)
        visibilities = np.concatenate(list_of_visibilities, axis=0)
        sigma = np.concatenate(list_of_sigma, axis=0)
    else:
        frequencies = np.average(list_of_frequencies, axis=0)
        uv_wavelengths = np.concatenate(list_of_uv_wavelengths, axis=1)
        visibilities = np.concatenate(list_of_visibilities, axis=1)
        sigma = np.concatenate(list_of_sigma, axis=1)

    return frequencies, uv_wavelengths, visibilities, sigma


def load_cube_data_weights(settings):
    """Load optional CASA weights for moment-0 noise propagation."""
    patterns = settings.get("data_patterns", {})
    weight_pattern = patterns.get("weights")
    if weight_pattern is None:
        return None

    directory = Path(settings["data_directory"])
    uids = settings["uids"]
    width = settings["width"]
    extra_context = settings.get("pattern_context", {})
    list_of_weights = []

    for uid in uids:
        fmt = {"uid": uid, "width": width, **extra_context}
        weights = load_exported_array(directory / weight_pattern.format(**fmt))
        list_of_weights.append(np.concatenate((weights[0], weights[1]), axis=1))

    if len(uids) == 1:
        return np.concatenate(list_of_weights, axis=0)
    return np.concatenate(list_of_weights, axis=1)


def build_tracer(settings, centre=None):
    mass_cfg = mass_model_for_settings(settings)
    if centre is None:
        centre_0 = mass_cfg["centre_0"]
        centre_1 = mass_cfg["centre_1"]
    else:
        centre_0, centre_1 = centre
    mass = al.mp.PowerLaw(
        centre=(centre_0, centre_1),
        ell_comps=(mass_cfg["elliptical_comps_0"], mass_cfg["elliptical_comps_1"]),
        einstein_radius=mass_cfg["einstein_radius"],
        slope=mass_cfg["slope"],
    )
    shear_swap = settings.get("swap_shear_components", True)
    gamma_1 = mass_cfg["shear_elliptical_comps_1"] if shear_swap else mass_cfg["shear_elliptical_comps_0"]
    gamma_2 = mass_cfg["shear_elliptical_comps_0"] if shear_swap else mass_cfg["shear_elliptical_comps_1"]
    shear = al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)

    m3 = None
    if "multipole_m3_elliptical_comps_0" in mass_cfg and "multipole_m3_elliptical_comps_1" in mass_cfg:
        m3 = al.mp.PowerLawMultipole(
            m=3,
            centre=mass.centre,
            einstein_radius=mass.einstein_radius,
            slope=mass.slope,
            multipole_comps=(
                mass_cfg["multipole_m3_elliptical_comps_0"],
                mass_cfg["multipole_m3_elliptical_comps_1"],
            ),
        )

    m4 = None
    if "multipole_m4_elliptical_comps_0" in mass_cfg and "multipole_m4_elliptical_comps_1" in mass_cfg:
        m4 = al.mp.PowerLawMultipole(
            m=4,
            centre=mass.centre,
            einstein_radius=mass.einstein_radius,
            slope=mass.slope,
            multipole_comps=(
                mass_cfg["multipole_m4_elliptical_comps_0"],
                mass_cfg["multipole_m4_elliptical_comps_1"],
            ),
        )

    return al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=settings["redshift_lens"],
                mass=mass,
                shear=shear,
                multipole_m3=m3,
                multipole_m4=m4,
            ),
            al.Galaxy(redshift=settings["redshift_source"], light=al.LightProfile()),
        ]
    )


def build_tracer_from_instance(instance, settings):
    """Build a fixed-mass tracer with centre taken from a fit instance."""
    return build_tracer(settings, centre=lens_centre_from_instance(instance, settings))


def parametric_model_from_settings(settings, profile):
    """Build the source (+ optional lens) ``af.Collection`` for parametric fits."""
    mode = normalization_mode_from_settings(settings)
    priors_cfg = priors_for_normalization_mode(
        priors_cfg=settings["priors"],
        mode=mode,
        settings=settings,
    )
    source = source_model_from_profile(profile, priors_cfg)
    if free_lens_centre_from_settings(settings):
        return af.Collection(
            galaxies=af.Collection(
                lens=lens_galaxy_model_from_settings(settings, free_centre=True),
                source=source,
            )
        )
    return af.Collection(galaxies=af.Collection(source=source))


def run_from_settings(settings):
    validate_lensing_settings(settings)
    mode = validate_normalization_settings(settings)
    if requires_phase1(mode):
        raise ValueError(
            f"normalization_mode='{mode}' uses the two-phase pipeline. "
            "Run scripts/run_fit.py, which dispatches automatically."
        )

    af.conf.instance.push(new_path=settings.get("config_path", "./config"), output_path=settings["output_path"])

    frequencies, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    autolens_utils.resolve_image_plane_grid_in_settings(settings, uv_wavelengths)

    img_n_pixels, img_pixel_scale, _ = autolens_utils.image_plane_grid_from_settings(
        settings
    )
    image_plane_grid_3d = Grid3D.uniform(
        n_pixels=img_n_pixels,
        pixel_scale=img_pixel_scale,
        n_channels=len(frequencies),
    )

    kinms_grid_3d = autolens_utils.kinms_source_grid_3d(
        settings, n_channels=len(frequencies)
    )

    z_step_kms = spectral_utils.z_step_kms_from_data_frequencies(frequencies)

    frequencies_arr = np.squeeze(np.asarray(frequencies, dtype=float))
    if np.nanmax(frequencies_arr) < 1.0e4:
        frequencies_hz = frequencies_arr * 1.0e9
    else:
        frequencies_hz = frequencies_arr

    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms,
        frequencies_hz=frequencies_hz,
    )
    free_lens_centre = free_lens_centre_from_settings(settings)
    tracer = None if free_lens_centre else build_tracer(settings)

    modelname = settings["model_name"]
    if modelname == "GalPak":
        profile = profiles.GalPaK
        dataset_instance = None
    elif modelname == "KinMS":
        profile = profiles.kinMS
        dataset_instance = kinms_utils.make_instance_from_grid(
            grid_3d=kinms_grid_3d,
            z_step_kms=z_step_kms,
            attach_grid=True,
            disk_thick=kinms_utils.disk_scale_height_arcsec_from_settings(settings),
        )
    else:
        raise ValueError(f"Unsupported model_name: {modelname}")

    mask_3d = Mask3D.unmasked(
        n_channels=image_plane_grid_3d.n_channels,
        shape_2d=image_plane_grid_3d.shape_2d,
        pixel_scales=image_plane_grid_3d.pixel_scales,
    )
    masked_dataset = MaskedDataset(dataset=dataset, mask_3d=mask_3d, instance=dataset_instance)
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=masked_dataset.uv_wavelengths,
        mask_3d=masked_dataset.mask_3d,
        settings=settings,
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

    model = parametric_model_from_settings(settings, profile)

    search = build_search_from_settings(settings["search"])
    result = search.fit(model=model, analysis=analysis_instance)

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

