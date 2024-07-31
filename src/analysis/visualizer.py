import os, sys
import numpy as np
import matplotlib.pyplot as plt

# NOTE:
import autofit as af

# NOTE:
from src.utils import (
    plot_utils as plot_utils,
    autolens_utils as autolens_utils,
)

# ---------------------------------------------------------------------------- #
# HACK:
# ---------------------------------------------------------------------------- #
class VisualizerAbstract:

    def __init__(
        self,
        masked_dataset,
        transformers,
        directory=None,
    ):

        self.transformers = transformers

        # NOTE:
        self.dirty_cube = autolens_utils.dirty_cube_from(
            visibilities=masked_dataset.data,
            transformers=transformers,
        )

        # NOTE:
        self.directory = directory


    def update(
        self,
        directory,
    ):

        if self.directory is None:
            self.directory = os.path.join(directory, "fit_dataset")
            if not os.path.isdir(self.directory):
                os.system(
                    "mkdir -p {}".format(self.directory)
                )
        else:
            raise NotImplementedError()

        # NOTE:
        self.visualize_data()


    def visualize_data(
        self,
    ):
        figure, axes = plot_utils.plot_cube(
            cube=self.dirty_cube,
        )
        plt.savefig(
            os.path.join(self.directory, "data.png")
        )
        plt.clf()


    def visualize(
        self,
        model_data,
        during_analysis=True,
    ):

        dirty_model_cube = autolens_utils.dirty_cube_from(
            visibilities=model_data,
            transformers=self.transformers,
        )

        # NOTE:
        residuals = self.dirty_cube - dirty_model_cube

        # NOTE:
        figure, axes = plot_utils.plot_cube(
            cube=dirty_model_cube,
        )
        plt.savefig(
            os.path.join(self.directory, "model.png")
        )
        plt.clf()

        # NOTE:
        figure, axes = plot_utils.plot_cube(
            cube=residuals,
        )
        plt.savefig(
            os.path.join(self.directory, "residuals.png")
        )
        plt.clf()
# ---------------------------------------------------------------------------- #
# END
# ---------------------------------------------------------------------------- #

"""
class Visualizer(af.Visualizer):

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.DirectoryPaths,
        model: af.AbstractPriorModel,
    ):
        print("----------------")
        print("----------------")
        print("calling \'visualize_before_fit\' from Visualizer class")
        print("----------------")
        print("----------------")

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
        print("----------------")
        print("----------------")
        print("calling \'visualize\' from Visualizer class")
        print("----------------")
        print("----------------")

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
"""
