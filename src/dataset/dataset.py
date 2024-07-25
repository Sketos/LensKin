import os, sys
import numpy as np

# NOTE:
import autolens as al

# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "{}/tutorials/autofit/tutorial_13"
sys.path.append(path)

from src.grid.grid import (
    Grid3D,
)
from src.mask.mask import (
    Mask3D,
)
# ---------------------------------------------------------------------------- #


# NOTE:
def make_kinms_instance(
    n_pixels: np.int,
    pixel_scale: np.float64,
    n_channels: np.int,
    z_step_kms: np.float64,
    nSamps=5e6,
):

    # NOTE
    try:
        from kinms import KinMS
    except:
        raise NotImplementedError()

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


# NOTE:
def make_instance_from_masked_dataset(
    masked_dataset,
    nSamps=5e6,
):

    return make_kinms_instance(
        n_pixels=masked_dataset.n_pixels,
        pixel_scale=masked_dataset.pixel_scale,
        n_channels=masked_dataset.n_channels,
        z_step_kms=masked_dataset.z_step_kms,
        nSamps=nSamps,
    )

# NOTE: PyAutoArray
def grid_from_mask(mask_3d: Mask3D):

    if al.__version__ == "0.45.0":
        return al.Grid.uniform(
            shape_2d=mask_3d.shape_2d,
            pixel_scales=mask_3d.pixel_scales,
            sub_size=mask_3d.sub_size
        )
    else:
        raise NotImplementedError()

class Dataset:
    def __init__(
        self,
        uv_wavelengths: np.ndarray,
        visibilities: np.ndarray,
        noise_map: np.ndarray,
        z_step_kms: np.float=None,
        redshift_source: np.float=2.0
    ):

        # NOTE: shape = (n_c, n_v, 2)
        self.uv_wavelengths = uv_wavelengths

        # NOTE:
        if visibilities.shape[0] == self.uv_wavelengths.shape[0]:
            self.visibilities = visibilities

        # NOTE: shape = (n_c, n_v, 2)
        self.noise_map = noise_map

        # NOTE:
        self.z_step_kms = z_step_kms

        # NOTE:
        self.redshift_source = redshift_source

    @property
    def shape(self):
        return visibilities.shape


class MaskedDataset:
    def __init__(
        self,
        dataset: Dataset,
        mask_3d: Mask3D=None,
        uv_mask: np.ndarray=None,
        grid_2d_dx: np.float=0.0, # NOTE: DELETE
        grid_2d_dy: np.float=0.0, # NOTE: DELETE
        condition=False,
    ):

        self.dataset = dataset

        # NOTE:
        self.mask_3d = mask_3d
        grid_2d = grid_from_mask(
            mask_3d=self.mask_3d
        )
        self.grid_3d = Grid3D(
            grid_2d=grid_2d,
            n_channels=mask_3d.n_channels
        )

        # # NOTE: This does not currently work
        # if grid_2d_dy != 0.0:
        #     self.grid_3d.grid_2d[..., 0] += grid_2d_dy
        # if grid_2d_dx != 0.0:
        #     self.grid_3d.grid_2d[..., 1] += grid_2d_dx

        # NOTE:
        self.uv_wavelengths = dataset.uv_wavelengths

        # NOTE:
        self.visibilities = dataset.visibilities
        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            self.uv_mask = uv_mask
        self.uv_mask[self.mask_3d.z_mask, ...] = True

        # self.uv_mask_real_and_imag_averaged = np.full(
        #     shape=self.uv_mask.shape[:-1],
        #     fill_value=False
        # )

        # NOTE:
        self.noise_map = dataset.noise_map

        # self.noise_map_real_and_imag_averaged = np.average(
        #     a=self.noise_map, axis=-1
        # )

        # NOTE:
        self.z_step_kms = dataset.z_step_kms

        # NOTE:
        self.redshift_source = dataset.redshift_source

        # NOTE:
        if condition:
            self.instance = make_instance_from_masked_dataset(
                masked_dataset=self,
            )
            # self.x = np.logspace(
            #     np.log10(self.pixel_scale / 5.0), np.log10(self.n_pixels * self.pixel_scale * 2.0), 10000
            # )
        else:
            self.instance = None
            #self.x = None


    @property
    def data(self):
        return self.visibilities

    @property
    def sigma(self):
        return self.noise_map

    @property
    def z_mask(self):
        return self.mask_3d.z_mask

    @property
    def n_channels(self):
        return self.mask_3d.n_channels

    @property
    def n_pixels(self):
        return self.mask_3d.n_pixels

    @property
    def pixel_scale(self):
        return self.mask_3d.pixel_scale
