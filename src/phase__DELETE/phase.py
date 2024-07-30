import os, sys
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import autofit as af
import autolens as al

# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "/tutorials/autofit/tutorial_13"
sys.path.append(path)

from src.dataset.dataset import (
    Dataset,
    MaskedDataset
)
from src.phase.result import (
    Result,
)
from src.phase.analysis import (
    Analysis,
)
# ---------------------------------------------------------------------------- #

# def reshape_array(array):
#
#     return array.reshape(
#         -1,
#         array.shape[-1]
#     )

# NOTE: PyAutoArray
def transformers_from(masked_dataset, transformer_class):

    transformers = []
    for i in range(masked_dataset.uv_wavelengths.shape[0]):
        transformer = transformer_class(
            uv_wavelengths=masked_dataset.uv_wavelengths[i],
            grid=masked_dataset.grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    return transformers


class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        profiles,
        tracer,
        non_linear_class=af.MultiNest,
        transformer_class=al.TransformerFINUFFT
    ):

        super().__init__(
            paths=paths,
            non_linear_class=non_linear_class,
        )

        self.profiles = profiles # NOTE: autofit.mapper.prior_model.collection.CollectionPriorModel

        # NOTE:
        self.tracer = tracer

        # NOTE:
        self.transformer_class = transformer_class


    @property
    def phase_folders(self):
        return self.optimizer.phase_folders


    def run(
        self,
        dataset: Dataset,
        mask_3d
    ):

        # NOTE:
        analysis = self.make_analysis(
            dataset=dataset,
            mask_3d=mask_3d
        )

        # NOTE:
        result = self.run_analysis(analysis=analysis)

        return self.make_result(
            result=result,
            analysis=analysis
        )

    def make_analysis(self, dataset, mask_3d):

        # NOTE:
        condition = np.any([
            profile.name == "kinMS" for profile in self.profiles
        ])
        masked_dataset = MaskedDataset(
            dataset=dataset,
            mask_3d=mask_3d,
            condition=condition,
        )

        # NOTE:
        transformers = transformers_from(
            masked_dataset=masked_dataset, transformer_class=self.transformer_class
        )

        return Analysis(
            masked_dataset=masked_dataset,
            transformers=transformers,
            tracer=self.tracer,
            image_path=self.optimizer.paths.image_path
        )

    def make_result(self, result, analysis):

        return self.Result(
            instance=result.instance,
            likelihood=result.likelihood,
            previous_model=result.model,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            output=result.output,
        )
