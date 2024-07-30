import os, sys
import numpy as np
import matplotlib.pyplot as plt


# NOTE:
from astropy import (
    units,
    constants
)
from astropy.io import fits


# NOTE:
import autofit as af#;print(af.__version__);exit()
af.conf.instance.push(
    new_path="./config", output_path="./output"
)
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# NOTE:
from src.dataset.dataset import (
    Dataset,
    MaskedDataset,
)
from src.grid.grid import (
    Grid3D,
)
from src.mask.mask import (
    Mask3D,
)
from src.model import (
    profiles,
)
from src.utils import (
    casa_utils as casa_utils,
    spectral_utils as spectral_utils,
    plot_utils as plot_utils,
    analysis_utils as analysis_utils,
    autolens_utils as autolens_utils
)
from src.analysis import (
    analysis,
)

# ============================================================================ #


def load_uv_wavelengths(
    n_channels,
    central_frequency=260.0 * units.GHz,
    filename="./uv.fits"
):

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
#n_channels = 64

lens_redshift = 0.5
source_redshift = 2.0

# NOTE:
central_frequency = 260.0 * units.GHz

# NOTE:
transformer_class = al.TransformerNUFFT
#transformer_class = al.TransformerDFT

if __name__ == "__main__":

    # NOTE:
    uv_wavelengths, frequencies = load_uv_wavelengths(
        n_channels=n_channels, central_frequency=central_frequency
    )

    # NOTE:
    z_step_kms = spectral_utils.compute_z_step_kms(
        frequencies=frequencies * units.Hz,
        frequency_0=central_frequency.to(units.Hz)
    )

    # NOTE: ...
    grid_3d = Grid3D.uniform(
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        n_channels=uv_wavelengths.shape[0]
    )

    # NOTE:
    model_default = profiles.GalPaK(
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
    cube = model_default.profile_cube_from_grid(
        grid_3d=grid_3d,
        z_step_kms=z_step_kms
    )
    # figure, axes = plot_utils.plot_cube(
    #     cube=cube,
    # )
    # plt.show()
    # exit()

    # NOTE:
    lens_mass_profile = al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        einstein_radius=1.0,
        slope=2.0
    )
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=lens_redshift,
                mass=lens_mass_profile,
            ),
            al.Galaxy(
                redshift=source_redshift,
                light=al.LightProfile()
            )
        ]
    )

    # NOTE:
    lensed_cube = analysis_utils.lensed_cube_from_tracer(
        tracer=tracer,
        grid=grid_3d.grid_2d,
        cube=cube
    )
    # figure, axes = plot_utils.plot_cube(
    #     cube=lensed_cube,
    # )
    # # for i in range(axes.shape[0]):
    # #     for j in range(axes.shape[1]):
    # #         axes[i, j].contour(profile_image.in_2d, colors="w")
    # plt.show()
    # exit()

    # NOTE:
    mask_3d = Mask3D.unmasked(
        n_channels=grid_3d.n_channels,
        shape_2d=grid_3d.shape_2d,
        pixel_scales=grid_3d.pixel_scales,
    )

    # NOTE:
    transformers = autolens_utils.transformers_from(
        uv_wavelengths=uv_wavelengths,
        mask_3d=mask_3d,
        transformer_class=transformer_class,
    )

    # NOTE:
    filename_visibilities = "./visibilities.fits"
    if not os.path.isfile(filename_visibilities):
        visibilities = autolens_utils.visibilities_from_transformers_and_cube(
            cube=lensed_cube, transformers=transformers, shape=uv_wavelengths.shape
        )
        # fits.writeto(
        #     filename_visibilities, data=visibilities
        # )
    else:
        visibilities = fits.getdata(filename=filename_visibilities)

    # NOTE:
    scale = 5.0 * 10**-1.0
    filename_noise = "./noise.fits"
    if not os.path.isfile(filename_noise):
        noise = np.random.normal(
            loc=0.0, scale=scale, size=visibilities.shape
        )
        # fits.writeto(
        #     filename_noise, data=noise
        # )
    else:
        noise = fits.getdata(filename=filename_noise)
    noise_map = scale * np.ones(shape=visibilities.shape)

    """# NOTE:
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(
        noise[int(grid_3d.n_channels / 2.0), :, 0],
        noise[int(grid_3d.n_channels / 2.0), :, 1],
        linestyle="None",
        marker=".",
        color="black",
    )
    axes[0].plot(
        visibilities[int(grid_3d.n_channels / 2.0), :, 0],
        visibilities[int(grid_3d.n_channels / 2.0), :, 1],
        linestyle="None",
        marker=".",
        color="r",
    )
    axes[1].plot(
        visibilities[int(grid_3d.n_channels / 2.0), :, 0] + noise[int(grid_3d.n_channels / 2.0), :, 0],
        visibilities[int(grid_3d.n_channels / 2.0), :, 1] + noise[int(grid_3d.n_channels / 2.0), :, 1],
        linestyle="None",
        marker=".",
        color="b",
    )
    plt.show()
    exit()
    """

    # NOTE:
    dataset = Dataset(
        uv_wavelengths=uv_wavelengths,
        visibilities=np.add(
            visibilities, noise
        ),
        noise_map=noise_map,
        z_step_kms=z_step_kms
    )
    # dirty_cube = autolens_utils.dirty_cube_from(
    #     visibilities=dataset.visibilities,
    #     transformers=transformers,
    # )
    # figure, axes = plot_utils.plot_cube(
    #     cube=dirty_cube,
    # )
    # # for i in range(axes.shape[0]):
    # #     for j in range(axes.shape[1]):
    # #         axes[i, j].contour(profile_image.in_2d, colors="w")
    # plt.show()
    # exit()

    # ============================================================================ #
    # ============================================================================ #

    # NOTE:
    # mask_3d = Mask3D.unmasked(
    #     n_channels=grid_3d.n_channels,
    #     shape_2d=grid_3d.shape_2d,
    #     pixel_scales=grid_3d.pixel_scales,
    # )
    masked_dataset = MaskedDataset(
        dataset=dataset,
        mask_3d=mask_3d,
    )
    analysis_lens_instance = analysis.Analysis(
        masked_dataset=masked_dataset,
        transformers=transformers,
        tracer=tracer,
    )

    # --- #
    # NOTE:
    # --- #
    """
    model = af.Model(profiles.GalPaK)
    array = np.linspace(0.0, 100.0, 50)
    likelihoods = np.zeros(shape=array.shape)
    for i, value in enumerate(array):
        instance = model.instance_from_vector(
            vector=[0.0, 0.2, grid_3d.n_channels / 2.0, 1.0, 0.2, 65.0, 90.0, 0.05, 250.0, value]
        )
        likelihoods[i] = analysis_lens_instance.log_likelihood_function(instance=instance)
    plt.figure()
    plt.plot(array, likelihoods, marker="o", color="b")
    plt.axvline(50.0, linestyle="--", color="black")
    plt.show()
    exit()
    """
    # --- #
    # END
    # --- #

    # NOTE:
    model = af.Collection(
        galaxies=af.Collection(
            source=af.Model(profiles.GalPaK)
        )
    )
    model.galaxies.source.maximum_velocity = model_default.maximum_velocity
    model.galaxies.source.z_centre = model_default.z_centre
    model.galaxies.source.intensity = model_default.intensity
    model.galaxies.source.effective_radius = model_default.effective_radius
    model.galaxies.source.inclination = model_default.inclination
    model.galaxies.source.phi = model_default.phi
    model.galaxies.source.turnover_radius = model_default.turnover_radius
    model.galaxies.source.maximum_velocity = model_default.maximum_velocity
    model.galaxies.source.velocity_dispersion = model_default.velocity_dispersion
    # model.galaxies.source.maximum_velocity = af.UniformPrior(
    #     lower_limit=0.0,
    #     upper_limit=600.0,
    # )

    # NOTE:
    name = "lens[fixed]_source[GalPaK]"
    if os.path.isdir(
        "./output/tutorial/{}".format(name)
    ):
        os.system(
            "rm -r ./output/tutorial/{}".format(name)
        )
    search = af.DynestyStatic(
        path_prefix=os.path.join("tutorial"),
        name=name,
        nlive=20,
        sample="rwalk",
        number_of_cores=1,
    )

    # NOTE:
    result = search.fit(model=model, analysis=analysis_lens_instance)
