import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import autofit as af
import autolens as al

# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "/tutorials/autofit/tutorial_13"
sys.path.append(path)

from src.dataset.dataset import (
    MaskedDataset,
)
from src.utils import (
    analysis_utils,
)
from src.phase import (
    visualizer,
)
from src.fit import fit as f
# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "/utils"
sys.path.append(path)
import plot_utils as plot_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils
import autolens_utils.autolens_tracer_utils as autolens_tracer_utils
# ---------------------------------------------------------------------------- #

class UnknownException(Exception):
    pass


# NOTE: PyAutoArray - hack
class Image:
    def __init__(self, array_2d: np.ndarray):
        self.array_2d = array_2d

    @property
    def in_2d_binned(self):
        return self.array_2d


# NOTE: PyAutoArray
def visibilities_from_image_and_transformer(image, transformer):
    if al.__version__ == "0.45.0":
        return transformer.visibilities_from_image(
            image=Image(array_2d=image)
        )
    else:
        raise NotImplementedError()


# NOTE: ...
def model_data_from_cube_and_transformers(
    cube,
    transformers,
    shape,
    z_mask
):

    # NOTE:
    model_data = np.zeros(shape=shape)
    for i, transformer in enumerate(transformers):
        if not z_mask[i]:
            model_data[i] = visibilities_from_image_and_transformer(
                image=cube[i], transformer=transformer
            )

    return model_data


class Analysis(af.Analysis):
    def __init__(
        self,
        masked_dataset: MaskedDataset,
        transformers: list,
        tracer,
        image_path=None
    ):

        # NOTE:
        self.masked_dataset = masked_dataset

        # NOTE:
        self.transformers = transformers

        # NOTE: If lens is fixed ...
        self.tracer = tracer

        # NOTE:
        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset,
            transformers=self.transformers,
            image_path=image_path
        )

        # NOTE: ...
        self.n_tot = 0


    def fit(self, instance):

        # NOTE:
        t_i = time.time()
        model_data = self.model_data_from_instance(
            instance=instance
        )
        t_j = time.time()
        print(
            "It took t={} to execute the \'model_data_from_instance\'".format(t_j - t_i)
        )

        # NOTE:
        t_i = time.time()
        fit = self.fit_from_model_data(
            model_data=model_data
        )
        likelihood = fit.likelihood
        t_j = time.time()
        print(
            "It took t={} to execute the \'fit_from_model_data\'.".format(t_j - t_i)
        )

        # NOTE:
        print(
            "n = {}; likelihood = {}".format(self.n_tot, likelihood)
        )
        self.n_tot += 1

        # NOTE:
        if np.isnan(likelihood):
            raise af.exc.FitException
        else:
            return fit.likelihood


    def model_cube_from_instance(
        self,
        instance,
    ):

        return sum([
            profile.profile_cube_from_masked_dataset(
                masked_dataset=self.masked_dataset,
            )
            for profile in instance.profiles
        ])


    def model_data_from_instance(
        self,
        instance,
    ):

        """
        # ---------- #
        # TODO: The instance can be a mass (model) + source (model). If so, use
        # the mass (model) to create the tracer and the source (model) to create
        # the model_cube...
        # ---------- #
        # NOTE:
        for profile in instance.profiles:
            print(type(profile), isinstance(profile, al.mp.MassProfile))
            # if profile.has_mass_profile:
            #     print("OK", profile.mass_profiles)
        # galaxies = [
        #     profile for profile in instance.profiles if isinstance(profile, al.mp.MassProfile)
        # ]
        # print(galaxies)
        # galaxies.append(
        #     al.Galaxy(
        #         redshift=self.masked_dataset.redshift_source,
        #         light=al.lp.LightProfile()
        #     )
        # )
        # tracer = al.Tracer.from_galaxies(
        #     galaxies=galaxies
        # )
        exit()
        """
        # ---------- #
        # END
        # ---------- #


        # NOTE:
        # t_i = time.time()
        model_cube = self.model_cube_from_instance(
            instance=instance
        )
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'model_cube_from_instance\'".format(t_j - t_i)
        # )

        # NOTE:
        # t_i = time.time()
        lensed_model_cube = analysis_utils.lensed_cube_from_tracer(
            cube=model_cube,
            tracer=self.tracer,
            grid=self.masked_dataset.grid_3d.grid_2d,
            z_mask=self.masked_dataset.mask_3d.z_mask,
        )
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'lensed_cube_from_tracer\'".format(t_j - t_i)
        # )

        # NOTE: Debugging visualization
        # plot_utils.plot_cube(
        #     cube=lensed_model_cube,
        #     ncols=5,
        #     figsize=(6.75, 8.15),
        #     show=False
        # )
        # # exit()

        # NOTE:
        # t_i = time.time()
        model_data = model_data_from_cube_and_transformers(
            cube=lensed_model_cube,
            transformers=self.transformers,
            shape=self.masked_dataset.data.shape,
            z_mask=self.masked_dataset.z_mask
        )
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'model_data_from_cube_and_transformers\'".format(t_j - t_i)
        # )

        # NOTE: Debugging visualization
        # plot_utils.plot_cube(
        #     cube=autolens_plot_utils.dirty_cube_from_visibilities(
        #         visibilities=model_data,
        #         transformers=self.transformers,
        #         shape=self.masked_dataset.grid_3d.shape_3d
        #     ),
        #     ncols=5,
        #     figsize=(6.75, 8.15),
        # )
        # exit()

        return model_data


    def fit_from_model_data(self, model_data):

        return f.DatasetFit(
            masked_dataset=self.masked_dataset,
            model_data=model_data
        )


    def visualize(self, instance, during_analysis):

        model_data = self.model_data_from_instance(
            instance=instance
        )
        fit = self.fit_from_model_data(
            model_data=model_data
        )

        self.visualizer.visualize_fit(
            fit=fit,
            during_analysis=during_analysis
        )
