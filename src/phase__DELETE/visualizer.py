import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_13".format(
        os.environ["GitHub"]
    )
)

from src.plot import dataset_plots, fit_plots


class AbstractVisualizer:
    def __init__(self, image_path):

        self.image_path = image_path


class Visualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, transformers, image_path):

        super().__init__(image_path)

        self.masked_dataset = masked_dataset

        self.transformers = transformers

        dataset_plots.data(
            masked_dataset=masked_dataset,
            transformers=self.transformers,
            output_filename="data",
            output_path=self.image_path,
            output_format="png",
        )

    def visualize_data(self, fit, during_analysis, output_format="png"):

        fit_plots.data(
            fit=fit,
            transformers=self.transformers,
            output_filename="fit_data",
            output_path=self.image_path,
            output_format=output_format,
        )

    def visualize_fit(self, fit, during_analysis, output_format="png"):

        fit_plots.model_data(
            fit=fit,
            transformers=self.transformers,
            output_filename="fit_model_data",
            output_path=self.image_path,
            output_format=output_format,
        )

        fit_plots.residual_map(
            fit=fit,
            transformers=self.transformers,
            output_filename="fit_residual_map",
            output_path=self.image_path,
            output_format=output_format,
        )

        # if not during_analysis:
        #
        #     fit_plots.normalized_residual_map(
        #         fit=fit,
        #         output_filename="fit_normalized_residual_map",
        #         output_path=self.image_path,
        #         output_format="png",
        #     )
