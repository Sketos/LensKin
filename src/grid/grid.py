import os, sys
import numpy as np

# NOTE:
import autolens as al


def func(grid_2d):

    return grid_2d


class AbstractGrid3D:
    def __init__(self):
        pass

class Grid3D(AbstractGrid3D):
    def __init__(
        self,
        grid_2d,
        n_channels: np.int
    ):

        # NOTE: PyAutoArray obj
        self.grid_2d = func(grid_2d=grid_2d)

        # NOTE:
        self.n_channels = n_channels


    @property
    def shape_2d(self):
        if hasattr(self.grid_2d, "shape_2d"):
            return self.grid_2d.shape_2d
        else:
            raise NotImplementedError()


    @property
    def shape_3d(self):
        return (self.n_channels, ) + self.shape_2d


    @property
    def pixel_scale(self):
        if hasattr(self.grid_2d, "pixel_scale"):
            return self.grid_2d.pixel_scale
        else:
            raise NotImplementedError()


    @property
    def pixel_scales(self):
        if hasattr(self.grid_2d, "pixel_scales"):
            return self.grid_2d.pixel_scales
        else:
            raise NotImplementedError()


    @property
    def sub_size(self):
        if hasattr(self.grid_2d, "sub_size"):
            return self.grid_2d.sub_size
        else:
            raise NotImplementedError()


    @classmethod
    def manual(
        cls,
        n_pixels: np.int,
        pixel_scale: np.float,
        n_channels: np.int,
    ) -> "Grid3D":

        if al.__version__ == "0.45.0":
            grid_2d = al.Grid.uniform(
                shape_2d=(
                    n_pixels,
                    n_pixels
                ),
                pixel_scales=(
                    pixel_scale,
                    pixel_scale
                ),
                sub_size=1
            )
        else:
            raise NotImplementedError()

        return Grid3D(
            grid_2d=grid_2d,
            n_channels=n_channels
        )
