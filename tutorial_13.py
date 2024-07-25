import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from astropy import (
    units,
    constants
)
from astropy.io import fits


def paths(autolens_version, cosma_server="7"):

    config_path = "./config_{}".format(
        autolens_version
    )
    if os.environ["HOME"].startswith("/cosma"):
        output_path = "{}/tutorials/autofit/tutorial_13/output".format(
            os.environ["COSMA{}_DATA_host".format(cosma_server)]
        )
    else:
        output_path = "./output"

    return config_path, output_path

autolens_version = "0.45.0"
config_path, output_path = paths(
    autolens_version=autolens_version
)

import autofit as af
af.conf.instance = af.conf.Config(
    config_path=config_path,
    output_path=output_path
)
import autolens as al
import autolens.plot as aplt


from src.grid.grid import Grid3D
from src.mask.mask import Mask3D, z_mask_from_zmin_and_zmax
from src.dataset.dataset import Dataset, \
                                MaskedDataset
from src.model import profiles
from src.phase import phase


sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils
import casa_utils as casa_utils
import spectral_utils as spectral_utils

import interferometry_utils.load_utils as interferometry_load_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
import autolens_utils.autolens_tracer_utils as autolens_tracer_utils


class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def in_1d_binned(self):
        return np.ndarray.flatten(self.array_2d)

    @property
    def in_2d_binned(self):
        return self.array_2d


# def load_uv_wavelengths():
#
#     filename = "{}/uv_wavelengths.fits".format(
#         os.path.dirname(
#             os.path.realpath(__file__)
#         )
#     )
#
#     if not os.path.isfile(filename):
#         raise IOError(
#             "The file {} does not exist".format(filename)
#         )
#
#     u_wavelengths, v_wavelengths = interferometry_load_utils.load_uv_wavelengths_from_fits(
#         filename=filename
#     )
#
#     uv_wavelengths = np.stack(
#         arrays=(u_wavelengths, v_wavelengths),
#         axis=-1
#     )
#
#     return uv_wavelengths

def load_uv_wavelengths(n_channels, central_frequency=260.0 * units.GHz, filename="./uv.fits"):

    frequencies = casa_utils.generate_frequencies(
        central_frequency=central_frequency,
        n_channels=n_channels,
        bandwidth=2.0 * units.GHz
    )

    uv = fits.getdata(filename=filename)

    uv_wavelengths = casa_utils.convert_uv_coords_from_meters_to_wavelengths(
        uv=uv,
        frequencies=frequencies
    )

    return uv_wavelengths, frequencies

# n_pixels = 256
# pixel_scale = 0.025
n_pixels = 64
pixel_scale = 0.1

n_channels = 16

lens_redshift = 0.5
source_redshift = 2.0


def spectrum_from(cube):

    spectrum = np.zeros(shape=(cube.shape[0], ))
    for i in range(cube.shape[0]):
        spectrum[i] = np.sum(cube[i, :, :])

    return spectrum


#transformer_class = al.TransformerFINUFFT
transformer_class = al.TransformerNUFFT

if __name__ == "__main__":

    central_frequency = 260.0 * units.GHz

    uv_wavelengths, frequencies = load_uv_wavelengths(
        central_frequency=central_frequency, n_channels=n_channels
    )

    z_step_kms = spectral_utils.compute_z_step_kms(
        frequencies=frequencies * units.Hz,
        frequency_0=central_frequency.to(units.Hz)
    )

    grid_3d = Grid3D(
        grid_2d=al.Grid.uniform(
            shape_2d=(
                n_pixels,
                n_pixels
            ),
            pixel_scales=(
                pixel_scale,
                pixel_scale
            ),
            sub_size=1
        ),
        n_channels=uv_wavelengths.shape[0]
    )

    model = profiles.Kinematical(
        centre=(0.0, 0.2),
        z_centre=grid_3d.n_channels / 2.0,
        intensity=1.0,
        effective_radius=0.2,
        inclination=65.0,
        phi=90.0,
        turnover_radius=0.05,
        maximum_velocity=250.0,
        velocity_dispersion=50.0
    )
    cube = model.profile_cube_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms
    )
    #plot_utils.plot_cube(cube=cube, ncols=16);exit()
    #exit()

    lens_mass_profile = al.mp.EllipticalPowerLaw(
        centre=(0.0, 0.0),
        axis_ratio=0.75,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0
    )
    tracer = al.Tracer.from_galaxies(
        galaxies=[
            al.Galaxy(
                redshift=lens_redshift,
                mass=lens_mass_profile,
            ),
            al.Galaxy(
                redshift=source_redshift,
                light=al.lp.LightProfile()
            )
        ]
    )

    # NOTE:
    lensed_cube = autolens_tracer_utils.lensed_cube_from_tracer(
        tracer=tracer,
        grid=grid_3d.grid_2d,
        cube=cube
    )
    # axes = plot_utils.plot_cube(
    #     cube=lensed_cube, ncols=16, figsize=(16, 6), show=False, return_axes=True
    # )
    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         axes[i, j].contour(profile_image.in_2d, colors="w")
    # plt.show()
    # exit()

    # NOTE:
    transformers = []
    for i in range(grid_3d.n_channels):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    # NOTE:
    visibilities = np.zeros(
        shape=uv_wavelengths.shape
    )
    for i in range(visibilities.shape[0]):
        visibilities[i] = transformers[i].visibilities_from_image(
                image=Image(
                    array_2d=lensed_cube[i]
                )
            )

    # plot_utils.plot_cube(
    #     cube=autolens_plot_utils.dirty_cube_from_visibilities(
    #         visibilities=visibilities,
    #         transformers=transformers,
    #         shape=grid_3d.shape_3d
    #     ),
    #     ncols=8,
    # )
    # exit()


    noise_map = np.random.normal(
        loc=0.0, scale=1.0 * 10**-1.0, size=visibilities.shape
    )
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities, noise_map
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )



    # --------- #

    phase_1_name = "phase_1__tutorial_13"
    # os.system(
    #     "rm -r output/{}".format(phase_1_name)
    # )

    phase_folders = []

    # model = profiles.Kinematical(
    #     centre=(0.0, 0.0),
    #     z_centre=16.0,
    #     intensity=1.0,
    #     effective_radius=0.5,
    #     inclination=30.0,
    #     phi=50.0,
    #     turnover_radius=0.05,
    #     maximum_velocity=200.0,
    #     velocity_dispersion=50.0
    # )
    source_model_1 = af.PriorModel(profiles.Kinematical)


    source_model_1.centre = model.centre
    source_model_1.z_centre = model.z_centre
    source_model_1.intensity = model.intensity
    source_model_1.inclination = model.inclination
    source_model_1.phi = model.phi
    source_model_1.turnover_radius = model.turnover_radius
    source_model_1.effective_radius = model.effective_radius
    source_model_1.maximum_velocity = model.maximum_velocity

    # source_model_1.z_centre = af.GaussianPrior(
    #     mean=16.0,
    #     sigma=2.0
    # )
    # source_model_1.intensity = af.LogUniformPrior(
    #     lower_limit=10**-2.0,
    #     upper_limit=10**+2.0
    # )
    # source_model_1.maximum_velocity = af.UniformPrior(
    #     lower_limit=25.0,
    #     upper_limit=400.0
    # )
    source_model_1.velocity_dispersion = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=100.0
    )

    phase_1 = phase.Phase(
        phase_name=phase_1_name,
        phase_folders=phase_folders,
        profiles=af.CollectionPriorModel(
            source_model=source_model_1,
        ),
        tracer=tracer,
        transformer_class=transformer_class
    )

    phase_1.optimizer.const_efficiency_mode = True
    phase_1.optimizer.n_live_points = 50
    phase_1.optimizer.sampling_efficiency = 0.5
    phase_1.optimizer.evidence_tolerance = 100

    mask_3d = Mask3D.xy_unmasked_and_z_mask(
        shape_2d=grid_3d.shape_2d,
        z_mask=z_mask_from_zmin_and_zmax(
            shape=(grid_3d.n_channels, ),
            zmin=0,
            zmax=grid_3d.n_channels
        ),
        pixel_scales=grid_3d.pixel_scales,
        sub_size=grid_3d.sub_size
    )

    result_phase_1 = phase_1.run(
        dataset=dataset,
        mask_3d=mask_3d
    )

    # model_data = result_phase_1.analysis.model_data_from_instance(
    #     instance=result_phase_1.instance, use_mask=False
    # )
    # print(model_data.shape)
    # exit()
