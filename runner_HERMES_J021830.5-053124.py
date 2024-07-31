import os, sys
import numpy as np

# NOTE:
from astropy import (
    units,
    constants
)
from astropy.io import fits

# NOTE:
server = sys.argv[1]
if server == "cosma7":
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def output_path_from(server="local"):

    if server == "local":
        output_path = "./output"
    elif server == "cosma7":
        output_path = "/cosma7/data/dp004/dc-amvr1"

        # NOTE:
        output_path += "/PyLensKin/output/"
    else:
        raise NotImplementedError()

    return output_path

# NOTE:
import autofit as af
output_path = output_path_from(
    server=server
)
af.conf.instance.push(
    new_path="./config", output_path=output_path
)
import autolens as al

# ---------------------------------------------------------------------------- #

from src.grid.grid import (
    Grid3D,
)
from src.mask.mask import (
    Mask3D,
)
from src.dataset.dataset import (
    Dataset,
    MaskedDataset,
)
from src.model import (
    profiles,
)
from src.utils import (
    spectral_utils as spectral_utils,
    autolens_utils as autolens_utils,
)
from src.analysis import (
    analysis,
)

# ---------------------------------------------------------------------------- #


# NOTE:
n_pixels = 128
pixel_scale = 0.0390625

# NOTE:
redshift_lens = 0.5
redshift_source = 2.0


# NOTE:
if server == "local":
    directory = "/Volumes/MyPassport_red"
elif server == "cosma7":
    directory = "/cosma7/data/dp004/dc-amvr1/ALMA_archive_data"
directory += "/2023.1.01354.S_HERMES_J021830.5-053124"

if __name__ == "__main__":

    # NOTE:
    n_scan_combined = 5
    #n_scan_combined = 10
    print(
        "scans combined =", n_scan_combined
    )

    # NOTE:
    if True:
        list_of_uv_wavelengths = []
        list_of_visibilities = []
        list_of_sigma = []

        # NOTE:
        uids = [
            "A002_X118bf6a_Xf99f"
        ]

        # NOTE:
        width = 30
        for uid in uids:
            print("uid =", uid)

            # NOTE:
            filename = "frequencies_{}_HERMES_J021830.5-053124_spw_31_width_{}_contsub.fits".format(
                uid, width
            )
            frequencies = fits.getdata(
                filename=directory + "/" + filename
            )
            # print(frequencies)

            # NOTE:
            filename = "uv_wavelengths_{}_HERMES_J021830.5-053124_spw_31_width_{}_contsub.fits".format(
                uid, width
            )
            uv_wavelengths = fits.getdata(
                filename=directory + "/" + filename
            )
            uv_wavelengths_temp = np.concatenate(
                (uv_wavelengths, uv_wavelengths),
                axis=1
            )
            list_of_uv_wavelengths.append(uv_wavelengths_temp)

            # NOTE:
            filename = "visibilities_{}_HERMES_J021830.5-053124_spw_31_width_{}_contsub.fits".format(
                uid, width
            )
            visibilities = fits.getdata(
                filename=directory + "/" + filename
            )
            visibilities_temp = np.concatenate(
                (visibilities[0, :, :, :], visibilities[1, :, :, :]),
                axis=1
            )
            list_of_visibilities.append(visibilities_temp)

            # NOTE:
            filename = "sigma_from_visibilities_{}_HERMES_J021830.5-053124_spw_31_width_{}_with_scans_x{}_contsub.fits".format(
                uid, width, n_scan_combined
            )
            sigma = fits.getdata(
                filename=directory + "/" + filename
            )
            sigma_temp = np.concatenate(
                (sigma[0, :, :, :], sigma[1, :, :, :]),
                axis=1
            )
            list_of_sigma.append(sigma_temp)
        if True: #if len(uids) > 1:
            uv_wavelengths = np.concatenate(
                list_of_uv_wavelengths,
                axis=0
            )
            visibilities = np.concatenate(
                list_of_visibilities,
                axis=0
            )
            sigma = np.concatenate(
                list_of_sigma,
                axis=0
            )
    #exit()

    # NOTE:
    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=len(frequencies)
    )

    # NOTE:
    redshift = 3.390
    frequency_0 = spectral_utils.observed_line_frequency_from_rest_line_frequency(
        frequency=1900.536900, redshift=redshift
    )
    velocities = spectral_utils.convert_frequencies_to_velocities(
        frequencies=frequencies * units.Hz.to(units.GHz), frequency_0=frequency_0, units=units.km / units.s
    )
    z_step_kms = abs(velocities[1] - velocities[0]);print(z_step_kms, "(km/s)")

    # NOTE:
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        noise_map=sigma,
        z_step_kms=z_step_kms
    )

    # ======================================================================== #

    #key = "parametric[1]"
    key = "inversion[0]"
    if key == "parametric[1]":
        raise NotImplementedError()
        # mass_centre_0 = 0.081
        # mass_centre_1 = -0.054
        # mass_einstein_radius = 0.509
        # mass_elliptical_comps_0 = -0.190
        # mass_elliptical_comps_1 = -0.018
        # mass_slope_mean = 2.0
        # shear_elliptical_comps_0 = -0.001
        # shear_elliptical_comps_1 = 0.075
    elif key == "inversion[0]":
        # NOTE: ML
        mass_centre_0 = 0.099
        mass_centre_1 = -0.067
        mass_einstein_radius = 0.514
        mass_elliptical_comps_0 = -0.184
        mass_elliptical_comps_1 = -0.052
        mass_slope = 2.0
        shear_elliptical_comps_0 = -0.001
        shear_elliptical_comps_1 = 0.067
    else:
        raise NotImplementedError()
    mass = al.mp.PowerLaw(
        centre=(
            mass_centre_0,
            mass_centre_1,
        ),
        ell_comps=(
            mass_elliptical_comps_0,
            mass_elliptical_comps_1,
        ),
        einstein_radius=mass_einstein_radius,
        slope=mass_slope,
    )
    shear = al.mp.ExternalShear(
        gamma_1=shear_elliptical_comps_0,
        gamma_2=shear_elliptical_comps_1,
    )
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=redshift_lens,
                mass=mass,
                shear=shear,
            ),
            al.Galaxy(
                redshift=redshift_source,
                light=al.LightProfile()
            )
        ]
    )

    # ======================================================================== #

    # NOTE:
    path_prefix = "HERMES_J021830.5-053124"

    # NOTE:
    name = "phase_{}".format(key)

    # NOTE:
    source = af.Model(profiles.GalPaK)
    model = af.Collection(
        galaxies=af.Collection(source=source)
    )
    # model.galaxies.maximum_velocity = af.UniformPrior(
    #     lower_limit=0.0,
    #     upper_limit=600.0,
    # )

    # NOTE:
    search = af.DynestyStatic(
        path_prefix=path_prefix,
        name=name,
        nlive=200,
        sample="rwalk",
        number_of_cores=1,
    )

    # NOTE:
    mask_3d = Mask3D.unmasked(
        n_channels=grid_3d.n_channels,
        shape_2d=grid_3d.shape_2d,
        pixel_scales=grid_3d.pixel_scales,
    )
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
    )
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=masked_dataset.uv_wavelengths,
        mask_3d=masked_dataset.mask_3d,
    )
    analysis_lens_instance = analysis.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
    )
    result = search.fit(
        model=model, analysis=analysis_lens_instance
    )

    # ======================================================================== #
