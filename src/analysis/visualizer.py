import os, sys
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import autofit as af

# ============================================================================ #
from src.utils import (
    plot_utils as plot_utils,
    autolens_utils as autolens_utils,
)
# ============================================================================ #


class Visualizer(af.Visualizer):

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.DirectoryPaths,
        model: af.AbstractPriorModel,
    ):

        # NOTE:
        if not os.path.isdir(paths.image_path):
            os.system("mkdir {}".format(paths.image_path))

        dirty_cube = autolens_utils.dirty_cube_from(
            visibilities=analysis.masked_dataset.data,
            transformers=analysis.transformers,
        )

        # NOTE:
        figure, axes = plot_utils.plot_cube(
            cube=dirty_cube,
        )
        plt.savefig(
            os.path.join(paths.image_path, "data.png")
        )
        plt.clf()


    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance,
        during_analysis,
    ):

        dirty_cube = autolens_utils.dirty_cube_from(
            visibilities=analysis.masked_dataset.data,
            transformers=analysis.transformers,
        )
        dirty_model_cube = autolens_utils.dirty_cube_from(
            visibilities=analysis.model_data_from_instance(instance=instance),
            transformers=analysis.transformers,
        )
        residuals = dirty_cube - dirty_model_cube

        # NOTE:
        figure, axes = plot_utils.plot_cube(
            cube=dirty_model_cube,
        )
        plt.savefig(
            os.path.join(paths.image_path, "model.png")
        )
        plt.clf()

        # NOTE:
        figure, axes = plot_utils.plot_cube(
            cube=residuals,
        )
        plt.savefig(
            os.path.join(paths.image_path, "residuals.png")
        )
        plt.clf()
