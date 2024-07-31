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

    #Visualizer = visualizer.Visualizer # NOTE: THIS IS NOT WORKING

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

        # NOTE:
        self.visualizer = visualizer.VisualizerAbstract(
            masked_dataset=self.masked_dataset,
            transformers=self.transformers,
        )


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
        # NOTE:
        # t_i = time.time()
        if np.any([
            isinstance(galaxy, al.mp.MassProfile) for galaxy in instance.galaxies
        ]):
            galaxies = [
                galaxy for galaxy in instance.galaxies if isinstance(galaxy, al.mp.MassProfile)
            ]
            galaxies.append(
                al.Galaxy(
                    redshift=self.masked_dataset.redshift_source,
                    light=al.LightProfile()
                )
            )
            tracer = al.Tracer(
                galaxies=galaxies
            )

            source_galaxies = [
                galaxy for galaxy in instance.galaxies if isinstance(galaxy, al.LightProfile)
            ]
            if len(source_galaxies) == 1:
                model_cube = self.model_cube_from_instance(
                    instance=source_galaxies[0],
                )
            else:
                raise NotImplementedError()
        else:
            model_cube = self.model_cube_from_instance(
                instance=instance
            )
        # t_j = time.time()
        # print(
        #     "It took t={} to execute the \'model_cube_from_instance\'".format(t_j - t_i)
        # )
        """

        model_cube = self.model_cube_from_instance(
            instance=instance
        )

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


    def visualize(self, paths, instance, during_analysis):

        if self.visualizer.directory is None:
            self.visualizer.update(directory=paths.image_path)

        model_data = self.model_data_from_instance(
            instance=instance
        )
        self.visualizer.visualize(
            model_data=model_data,
            during_analysis=during_analysis,
        )
