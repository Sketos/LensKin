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
from src.pipelines.lens_model import lens_centre_from_instance
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
    z_mask,
    primary_beam=None,
):

    return autolens_utils.visibilities_from_transformers_and_cube(
        cube=cube,
        transformers=transformers,
        shape=shape,
        z_mask=z_mask,
        primary_beam=primary_beam,
    )


class Analysis(af.Analysis):

    #Visualizer = visualizer.Visualizer # NOTE: THIS IS NOT WORKING

    def __init__(
        self,
        masked_dataset: MaskedDataset,
        transformers: list,
        tracer: al.Tracer = None,
        settings=None,
    ):
        super().__init__(use_jax=False)

        # NOTE:
        self.masked_dataset = masked_dataset

        # NOTE:
        self.transformers = transformers

        self.settings = settings

        # NOTE: Fixed tracer for fits with a frozen lens mass model.
        self.tracer = tracer
        if tracer is None and not self._uses_dynamic_lens_centre():
            raise ValueError(
                "Analysis requires a fixed tracer unless settings['free_lens_centre'] "
                "is true (or reconstruction.fix_lens is false for phase-1)."
            )

        self._flip_velocity_axis = False
        self._flip_kinms_y = True
        if self.settings is not None:
            self._flip_velocity_axis = bool(
                self.settings.get(
                    "flip_velocity_axis",
                    self.settings.get("lensing", {}).get("flip_velocity_axis", False),
                )
            )
            self._flip_kinms_y = bool(
                self.settings.get(
                    "flip_kinms_y_before_lensing",
                    self.settings.get("lensing", {}).get(
                        "flip_kinms_y_before_lensing", True
                    ),
                )
            )
        self._primary_beam = None
        if self.settings is not None:
            from src.utils.primary_beam import primary_beam_from_settings

            frequencies_hz = getattr(masked_dataset, "frequencies_hz", None)
            if frequencies_hz is not None:
                self._primary_beam = primary_beam_from_settings(
                    settings=self.settings,
                    frequencies_hz=frequencies_hz,
                    mask_2d=masked_dataset.mask_3d.mask_2d,
                )
            if self._primary_beam is not None:
                pb_min = float(self._primary_beam.min())
                print(
                    f"Primary beam enabled: min attenuation = {pb_min:.6f} "
                    f"(at field edge)"
                )

        print(
            "Analysis spectral/spatial conventions: "
            f"flip_velocity_axis={self._flip_velocity_axis}, "
            f"flip_kinms_y_before_lensing={self._flip_kinms_y} "
            f"(settings={'set' if self.settings is not None else 'None'})"
        )

        # NOTE:
        self.visualizer = visualizer.VisualizerAbstract(
            masked_dataset=self.masked_dataset,
            transformers=self.transformers,
        )

    def _uses_dynamic_lens_centre(self):
        if self.settings is None:
            return False
        from src.pipelines.lens_model import free_lens_centre_from_settings

        return free_lens_centre_from_settings(self.settings)

    def _tracer_for_instance(self, instance):
        if self._uses_dynamic_lens_centre() and hasattr(instance, "galaxies"):
            if hasattr(instance.galaxies, "lens"):
                from src.pipelines.runner import build_tracer

                return build_tracer(
                    self.settings,
                    centre=lens_centre_from_instance(instance, self.settings),
                )
        if self.tracer is not None:
            return self.tracer
        raise ValueError("No tracer available for this instance.")

    @staticmethod
    def _source_profile_from_instance(instance):
        if hasattr(instance, "galaxies"):
            if hasattr(instance.galaxies, "source"):
                return instance.galaxies.source
            profiles = [
                galaxy
                for galaxy in instance.galaxies
                if hasattr(galaxy, "profile_cube_from_masked_dataset")
            ]
            if len(profiles) == 1:
                return profiles[0]
            if len(profiles) > 1:
                raise NotImplementedError(
                    "Multiple kinematic source profiles in one instance."
                )
        if hasattr(instance, "profile_cube_from_masked_dataset"):
            return instance
        raise NotImplementedError(
            "Could not identify a kinematic source profile on the instance."
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
        likelihood = fit.likelihood
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
        source = self._source_profile_from_instance(instance)
        cube = source.profile_cube_from_masked_dataset(
            masked_dataset=self.masked_dataset,
        )
        # Test / convention flag: reverse spectral axis (velocity ↔ frequency order).
        if getattr(self, "_flip_velocity_axis", False):
            cube = np.ascontiguousarray(cube[::-1, :, :])
        return cube


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

        instance_obj = self.masked_dataset.instance
        source_grid_2d = None
        output_shape = None
        if instance_obj is not None and getattr(instance_obj, "grid_3d", None) is not None:
            source_grid_2d = instance_obj.grid_3d.grid_2d
            output_shape = self.masked_dataset.grid_3d.shape_2d

        flip_kinms_y = getattr(self, "_flip_kinms_y", True)

        # NOTE:
        # t_i = time.time()
        lensed_model_cube = analysis_utils.lensed_cube_from_tracer(
            cube=model_cube,
            tracer=self._tracer_for_instance(instance),
            grid=self.masked_dataset.grid_3d.grid_2d,
            z_mask=self.masked_dataset.mask_3d.z_mask,
            source_grid_2d=source_grid_2d,
            output_shape=output_shape,
            flip_kinms_y=flip_kinms_y,
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
            z_mask=self.masked_dataset.z_mask,
            primary_beam=self._primary_beam,
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
