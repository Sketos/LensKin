import os, sys
import numpy as np

# NOTE:
try:
    from kinms import KinMS
except:
    print("\'kinMS\' could not be imported")

# NOTE:
from src.grid.grid import (
    Grid3D,
)


class InstanceKinMS:

    def __init__(self, obj, x=None):

        self.obj = obj

        # NOTE:
        if x is None:
            self.x = np.logspace(
                np.log10(self.obj.cellSize / 5.0), np.log10(np.hypot(self.obj.xs, self.obj.ys)), 10000
            )
        else:
            self.x = x


    @property
    def instance(self):
        return self.obj



# NOTE:
def make_kinms_instance(
    n_pixels: int,
    pixel_scale: float,
    n_channels: int,
    z_step_kms: float,
    nSamps=5e6,
):

    return KinMS(
        xs=n_pixels * pixel_scale,
        ys=n_pixels * pixel_scale,
        vs=n_channels * z_step_kms,
        cellSize=pixel_scale,
        dv=z_step_kms,
        beamSize=None,
        cleanOut=True,
        nSamps=nSamps
    )


# # NOTE:
# def make_instance_from_masked_dataset(
#     masked_dataset,
#     nSamps=5e6,
# ):
#
#     return make_kinms_instance(
#         n_pixels=masked_dataset.n_pixels,
#         pixel_scale=masked_dataset.pixel_scale,
#         n_channels=masked_dataset.n_channels,
#         z_step_kms=masked_dataset.z_step_kms,
#         nSamps=nSamps,
#     )

def make_instance_from_grid(
    grid_3d: Grid3D,
    z_step_kms: float,
    nSamps=5e6,
):

    obj = make_kinms_instance(
        n_pixels=grid_3d.n_pixels,
        pixel_scale=grid_3d.pixel_scale,
        n_channels=grid_3d.n_channels,
        z_step_kms=z_step_kms,
        nSamps=nSamps,
    )

    return InstanceKinMS(
        obj=obj
    )
