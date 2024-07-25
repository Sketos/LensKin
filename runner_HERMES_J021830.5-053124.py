import os, sys
import numpy as np

from astropy import (
    units,
    constants
)
from astropy.io import fits

# NOTE:
server = "local"
#server = "cosma7"
if server == "cosma7":
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def paths(autolens_version="0.45.0", server="cosma7"):

    config_path = "./config_{}".format(
        autolens_version
    )
    if server == "local":
        output_path = "./output"
    elif server == "cosma7":
        output_path = "/cosma7/data/dp004/dc-amvr1"

        # NOTE:
        output_path += "/tutorials/autofit/tutorial_13/output"
    else:
        raise NotImplementedError()

    return config_path, output_path

# NOTE:
config_path, output_path = paths(server=server)

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autolens as al
import autolens.plot as aplt

# ---------------------------------------------------------------------------- #

from src.grid.grid import (
    Grid3D,
)
from src.mask.mask import (
    Mask3D,
    z_mask_from_zmin_and_zmax
)
from src.dataset.dataset import (
    Dataset,
    MaskedDataset,
)
from src.model import (
    profiles,
)
from src.phase import (
    phase,
)
from src.utils import (
    profile_utils,
    plot_utils,
)

# ---------------------------------------------------------------------------- #

path = os.environ["GitHub"] + "{}/utils"
sys.path.append(path)

import spectral_utils as spectral_utils
#import plot_utils as plot_utils
#import casa_utils as casa_utils
#
#import autolens_utils.autolens_plot_utils as autolens_plot_utils
#import autolens_utils.autolens_tracer_utils as autolens_tracer_utils

# ---------------------------------------------------------------------------- #

# NOTE:
class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_1d_binned(self):
        return np.ndarray.flatten(self.array_2d)

    @property
    def in_2d_binned(self):
        return self.array_2d





# NOTE:
n_pixels = 128
pixel_scale = 0.0390625
grid_2d = al.Grid.uniform(
    shape_2d=(
        n_pixels,
        n_pixels
    ),
    pixel_scales=(
        pixel_scale,
        pixel_scale
    ),
    sub_size=1
)

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
    grid_3d = Grid3D(
        grid_2d=grid_2d,
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

    # plot_utils.plot_dirty_cube(
    #     uv_wavelengths=dataset.uv_wavelengths,
    #     visibilities=dataset.visibilities,
    #     grid_3d=grid_3d,
    # )
    # exit()

    # ======================================================================== #

    #key = "parametric[1]"
    key = "inversion[0]"
    if key == "parametric[1]":
        mass_centre_0 = 0.081
        mass_centre_1 = -0.057
        mass_einstein_radius = 0.509
        mass_elliptical_comps_0 = -0.174
        mass_elliptical_comps_1 = 0.003
        mass_slope = 2.0
        shear_elliptical_comps_0 = -0.081
        shear_elliptical_comps_1 = -0.039
    elif key == "inversion[0]":
        # NOTE: ML
        mass_centre_0 = 0.100
        mass_centre_1 = -0.067
        mass_einstein_radius = 0.515
        mass_elliptical_comps_0 = -0.184
        mass_elliptical_comps_1 = -0.056
        mass_slope = 2.0
        shear_elliptical_comps_0 = 0.000
        shear_elliptical_comps_1 = 0.064
    else:
        raise NotImplementedError()
    mass_axis_ratio, mass_angle = profile_utils.axis_ratio_and_phi_from(
        elliptical_comps=(mass_elliptical_comps_0, mass_elliptical_comps_1)
    )
    shear_magnitude, shear_angle = profile_utils.shear_magnitude_and_phi_from(
        elliptical_comps=(shear_elliptical_comps_0, shear_elliptical_comps_1)
    )
    mass = al.mp.EllipticalPowerLaw(
        centre=(
            mass_centre_0,
            mass_centre_1,
        ),
        axis_ratio=mass_axis_ratio,
        phi=mass_angle,
        einstein_radius=mass_einstein_radius,
        slope=mass_slope,
    )
    shear = al.mp.ExternalShear(
        magnitude=shear_magnitude,
        phi=shear_angle,
    )
    tracer = al.Tracer.from_galaxies(
        galaxies=[
            al.Galaxy(
                redshift=redshift_lens,
                mass=mass,
                shear=shear,
            ),
            al.Galaxy(
                redshift=redshift_source,
                light=al.lp.LightProfile()
            )
        ]
    )

    # === #
    # NOTE: DELETE
    # === #
    # grid = al.Grid.uniform(
    #     shape_2d=(
    #         n_pixels,
    #         n_pixels
    #     ),
    #     pixel_scales=(
    #         pixel_scale,
    #         pixel_scale
    #     ),
    #     sub_size=1
    # )
    # lens_galaxy = al.Galaxy(
    #     redshift=redshift_lens,
    #     mass=mass,
    #     shear=shear,
    # )
    # deflections = lens_galaxy.deflections_from_grid(grid=grid).in_2d
    # deflections_y = deflections[:, :, 0]
    # deflections_x = deflections[:, :, 1]
    # directory_temp = "/Users/ccbh87/Desktop/GitHub/runners/autolens_2021.10.14.1/HERMES_J021830.5-053124/"
    # deflections_y_temp = fits.getdata(filename=directory_temp + "deflections_y_2d_from_lens_galaxy.fits")
    # deflections_x_temp = fits.getdata(filename=directory_temp + "deflections_x_2d_from_lens_galaxy.fits")
    # idx = deflections_y_temp == 0.0
    # deflections_y_temp[idx] = np.nan
    # deflections_y[idx] = np.nan
    # idx = deflections_x_temp == 0.0
    # deflections_x_temp[idx] = np.nan
    # deflections_x[idx] = np.nan
    # figure, axes = plt.subplots(nrows=2, ncols=3)
    # axes[0, 0].imshow(deflections_y)
    # axes[0, 1].imshow(deflections_y_temp)
    # axes[0, 2].imshow(deflections_y - deflections_y_temp)
    # axes[1, 0].imshow(deflections_x)
    # axes[1, 1].imshow(deflections_x_temp)
    # axes[1, 2].imshow(deflections_x - deflections_x_temp)
    # plt.show()
    # exit()
    # === #
    # END
    # === #


    # ======================================================================== #

    # NOTE:
    phase_1_name = "HERMES_J021830.5-053124/phase_{}_dev".format(key)
    os.system(
        "rm -r output/{}".format(phase_1_name)
    )

    # NOTE:
    phase_folders = []

    # # NOTE:
    # lens = al.GalaxyModel(
    #     redshift=redshift_lens,
    #     mass=al.mp.EllipticalPowerLaw,
    #     shear=al.mp.ExternalShear
    # )

    # NOTE:
    #source_model = af.PriorModel(profiles.Kinematical)
    source_model = af.PriorModel(profiles.kinMS)
    source_model.z_centre = af.GaussianPrior(
        mean=16.0,
        sigma=4.0
    )
    source_model.intensity = af.LogUniformPrior(
        lower_limit=10**-2.0,
        upper_limit=10**+2.0
    )
    source_model.phi = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=360.0,
    )
    source_model.maximum_velocity = af.UniformPrior(
        lower_limit=25.0,
        upper_limit=800.0
    )
    source_model.velocity_dispersion = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=200.0
    )

    phase_1 = phase.Phase(
        phase_name=phase_1_name,
        phase_folders=phase_folders,
        profiles=af.CollectionPriorModel(
            #lens=lens,
            source_model=source_model,
        ),
        tracer=tracer
    )

    phase_1.optimizer.const_efficiency_mode = True
    phase_1.optimizer.n_live_points = 200
    phase_1.optimizer.sampling_efficiency = 0.2
    phase_1.optimizer.evidence_tolerance = 0.5

    # NOTE:
    z_mask = z_mask_from_zmin_and_zmax(
        shape=(grid_3d.n_channels, ),
        zmin=0,
        zmax=grid_3d.n_channels
    )
    mask_3d = Mask3D.xy_unmasked_and_z_mask(
        z_mask=z_mask,
        shape_2d=grid_3d.shape_2d,
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size,
    )

    # NOTE:
    result_phase_1 = phase_1.run(
        dataset=dataset,
        mask_3d=mask_3d
    )
