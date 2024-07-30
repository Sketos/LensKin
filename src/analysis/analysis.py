import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import autofit as af
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# ---------------------------------------------------------------------------- #

from src.dataset.dataset import (
    MaskedDataset,
)
from src.utils import (
    analysis_utils as analysis_utils,
    autolens_utils as autolens_utils,
)
from src.fit import (
    fit,
)
from src.analysis import (
    visualizer,
)

# ---------------------------------------------------------------------------- #


# class UnknownException(Exception):
#     pass


# NOTE: ...
def model_data_from_cube_and_transformers(
    cube,
    transformers,
    shape,
    z_mask
):

    return autolens_utils.visibilities_from_transformers_and_cube(
        cube=cube,
        transformers=transformers,
        shape=shape,
        z_mask=z_mask,
    )


class Analysis(af.Analysis):

    Visualizer = visualizer.Visualizer

    def __init__(
        self,
        masked_dataset: MaskedDataset,
        transformers: list,
        tracer: al.Tracer = None,
    ):

        # NOTE:
        self.masked_dataset = masked_dataset

        # NOTE:
        self.transformers = transformers

        # NOTE: If lens is fixed ...
        if tracer is not None:
            self.tracer = tracer
        else:
            raise NotImplementedError()

        # # NOTE:
        # self.visualizer = visualizer.Visualizer(
        #     masked_dataset=self.masked_dataset,
        #     transformers=self.transformers,
        #     image_path=image_path
        # )


    def log_likelihood_function(self, instance):

        # NOTE:
        # t_i = time.time()
        model_data = self.model_data_from_instance(
            instance=instance
        )
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'model_data_from_instance\'".format(t_j - t_i)
        # )

        # NOTE:
        # t_i = time.time()
        fit = self.fit_from_model_data(
            model_data=model_data
        )
        likelihood = fit.likelihood;print("likelihood =", likelihood)
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'fit_from_model_data\'.".format(t_j - t_i)
        # )

        # NOTE:
        if np.isnan(likelihood):
            raise af.exc.FitException
        else:
            return fit.likelihood


    def model_cube_from_instance(
        self,
        instance,
    ) -> np.ndarray:
        # TODO: ...

        if isinstance(instance, al.LightProfile):
            return instance.profile_cube_from_masked_dataset(
                masked_dataset=self.masked_dataset,
            )
        elif hasattr(instance, "galaxies"):
            return sum([
                galaxy.profile_cube_from_masked_dataset(
                    masked_dataset=self.masked_dataset,
                )
                for galaxy in instance.galaxies
            ])
        else:
            raise NotImplementedError()


    def model_data_from_instance(
        self,
        instance,
    ) -> np.ndarray:

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

        return model_data


    def fit_from_model_data(self, model_data):

        return fit.DatasetFit(
            masked_dataset=self.masked_dataset,
            model_data=model_data
        )


    # def visualize(self, instance, during_analysis):
    #
    #     model_data = self.model_data_from_instance(
    #         instance=instance
    #     )
    #     fit = self.fit_from_model_data(
    #         model_data=model_data
    #     )
    #
    #     self.visualizer.visualize_fit(
    #         fit=fit,
    #         during_analysis=during_analysis
    #     )


    def visualize(self, paths, instance, during_analysis):
        print("----------------")
        print("----------------")
        print("----- HERE -----")
        print("----------------")
        print("----------------")

        # self.visualizer.visualize(
        #     analysis=self,
        #     paths=paths,
        #     instance=instance,
        #     during_analysis=during_analysis,
        # )
