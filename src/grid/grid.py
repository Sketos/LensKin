import os, sys
import numpy as np

# NOTE:
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# NOTE:
from src.mask.mask import (
    Mask3D,
)
from src.utils import (
    autoarray_utils as autoarray_utils,
)


def func(grid_2d):
    return grid_2d


class AbstractGrid3D:
    def __init__(self):
        pass


class Grid3D(AbstractGrid3D):
    def __init__(
        self,
        grid_2d: autoarray_utils.AutoArrayObj,
        n_channels: int
    ):

        # NOTE: PyAutoArray obj
        self.grid_2d = func(grid_2d=grid_2d)

        # NOTE:
        if n_channels == 1:
            raise NotImplementedError()
        else:
            self.n_channels = n_channels


    @property
    def shape_2d(self):
        if hasattr(self.grid_2d, "shape_native"):
            return self.grid_2d.shape_native
        else:
            raise NotImplementedError()


    @property
    def shape_3d(self):
        return (self.n_channels, ) + self.shape_2d


    @property
    def n_pixels(self):
        return self.shape_2d[0]


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


    @classmethod
    def uniform(
        cls,
        n_pixels: int,
        pixel_scale: float,
        n_channels: int,
    ) -> "Grid3D":

        if True: # al.__version__ == ???
            grid_2d = al.Grid2D.uniform(
                shape_native=(
                    n_pixels,
                    n_pixels
                ),
                pixel_scales=(
                    pixel_scale,
                    pixel_scale
                ),
            )
        else:
            raise NotImplementedError()

        return Grid3D(
            grid_2d=grid_2d,
            n_channels=n_channels
        )


def grid_from_mask(
    mask_3d: Mask3D
):

    return uniform(
        n_pixels=mask_3d.n_pixels,
        pixel_scale=mask_3d.pixel_scale,
        n_channels=mask_3d.n_channels,
    )
