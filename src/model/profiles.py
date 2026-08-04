import os, sys
import numpy as np

# NOTE:
import autofit as af
import autolens as al

from src.utils.analysis_utils import resample_image_to_shape
from src.utils import kinms_utils

# NOTE:
try:
    import galpak
except ImportError:  # pragma: no cover - optional dependency
    galpak = None
    print("'galpak' could not be imported")


# ============================================================================ #
# ============================================================================ #

class Abstract(al.LightProfile):
    def __init__(self):
        pass


class GalPaK(Abstract):

    def __init__(
        self,
        centre = (0.0, 0.0),
        z_centre: float = 0.0,
        intensity: float = 0.1,
        effective_radius: float = 1.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
    ):
        if galpak is None:
            raise ImportError(
                "GalPaK requires the 'galpak' package "
                "(pip install 'galpak==1.34.0')."
            )
        super(GalPaK, self).__init__()

        self.centre = centre
        self.z_centre = z_centre
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.inclination = inclination
        self.phi = phi
        self.turnover_radius = turnover_radius
        self.maximum_velocity = maximum_velocity
        self.velocity_dispersion = velocity_dispersion


    # NOTE:
    def convert_centre_from_arcsec_to_pixels(
        self,
        value,
        pixel_scale,
        n_pixels,
    ):
        return value / pixel_scale + n_pixels / 2.0 - 0.5


    # NOTE:
    def convert_radius_from_arcsec_to_pixels(
        self,
        value,
        pixel_scale,
    ):
        return value / pixel_scale


    # NOTE: ...
    def convert_parameters(
        self,
        grid_3d
    ):
        #start = time.time()

        # NOTE:
        #galpak = (x, y)
        #autolens_centre = (y, x)

        # --- #
        # NOTE:
        # --- #
        # for i, (name, value) in enumerate(
        #     self.__dict__.items()
        # ):
        #     print(i, ":", name, value)
        # exit()
        # --- #
        # END
        # --- #

        names = []

        # NOTE: ...
        converted_parameters = []
        for i, (name, value) in enumerate(
            self.__dict__.items()
        ):
            #print(name)
            if name not in ["id", "_assertions", "cls"]:

                if name == "centre":
                    names.append("centre_0")
                    names.append("centre_1")
                else:
                    names.append(name)

                # NOTE:
                if name == "centre":
                    for (i, sign) in zip([1, 0], [1.0, -1.0]):
                        converted_parameters.append(
                            self.convert_centre_from_arcsec_to_pixels(
                                value=sign * value[i],
                                pixel_scale=grid_3d.pixel_scale,
                                n_pixels=grid_3d.n_pixels,
                            )
                        )
                elif name in ["effective_radius", "turnover_radius"]:
                    converted_parameters.append(
                        self.convert_radius_from_arcsec_to_pixels(
                            value=value,
                            pixel_scale=grid_3d.pixel_scale,
                        )
                    )
                else:
                    converted_parameters.append(value)
        # end = time.time()
        # print(
        #     "It took t={} to convert parameters".format(end - start)
        # )

        # # NOTE:
        # print(
        #     "parameters (converted)", converted_parameters
        # )

        return converted_parameters


    # NOTE: ...
    def profile_cube_from_grid(
        self,
        grid_3d,
        z_step_kms: float,
        instance=None,
    ):

        # NOTE: ...
        model = galpak.DiskModel(
            flux_profile='exponential',
            thickness_profile="gaussian",
            rotation_curve='isothermal',
            dispersion_profile="thick"
        )
        galaxy = galpak.GalaxyParameters.from_ndarray(
            a=self.convert_parameters(grid_3d=grid_3d)
        )
        cube, _, _, _ = model._create_cube(
            galaxy=galaxy,
            shape=grid_3d.shape_3d,
            z_step_kms=z_step_kms,
            zo=self.z_centre
        )

        return cube.data


    # NOTE: ...
    def profile_cube_from_masked_dataset(self, masked_dataset):
        grid_3d = masked_dataset.grid_3d
        instance = masked_dataset.instance
        # Mode-2 attaches a source-plane grid on the instance (phase-1 bbox);
        # fall back to the image-plane mask grid for mode-1 GalPaK.
        if instance is not None and getattr(instance, "grid_3d", None) is not None:
            grid_3d = instance.grid_3d
        return self.profile_cube_from_grid(
            grid_3d=grid_3d,
            z_step_kms=masked_dataset.z_step_kms,
            instance=instance,
        )

# ============================================================================ #
# ============================================================================ #

class kinMS(Abstract):

    def __init__(
        self,
        centre = (0.0, 0.0),
        z_centre: float = 0.0,
        intensity: float = 0.1,
        effective_radius: float = 1.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
        vmax_black_hole: float = 0.0
    ):
        super(kinMS, self).__init__()

        self.centre = centre
        self.z_centre = z_centre
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.inclination = inclination
        self.phi = phi
        self.turnover_radius = turnover_radius
        self.maximum_velocity = maximum_velocity
        self.velocity_dispersion = velocity_dispersion
        self.vmax_black_hole = vmax_black_hole

    # NOTE:
    def convert_centre_from_arcsec_to_pixels(
        self,
        pixel_scale,
        n_pixels,
    ):

        centre_0_converted = n_pixels / 2.0 - self.centre[0] / pixel_scale
        centre_1_converted = n_pixels / 2.0 + self.centre[1] / pixel_scale
        return (
            centre_0_converted,
            centre_1_converted,
        )


    def make_model(self, instance):
        x = instance.x
        int_flux = instance.int_flux
        if int_flux is None:
            int_flux = self.__dict__["intensity"]

        sbprof = np.exp(-x / self.__dict__["effective_radius"])
        velprof = np.hypot(
            (2.0 * self.__dict__["maximum_velocity"] / np.pi)
            * np.arctan(x / self.__dict__["turnover_radius"]),
            self.__dict__["vmax_black_hole"] / np.sqrt(x),
        )

        cube = instance.obj.model_cube(
            inc=self.__dict__["inclination"],
            posAng=self.__dict__["phi"],
            intFlux=int_flux,
            gasSigma=self.__dict__["velocity_dispersion"],
            diskThick=getattr(instance, "disk_thick", 0.0) or 0.0,
            sbProf=sbprof,
            velProf=velprof,
            sbRad=x,
            velRad=x,
            inClouds=np.zeros((0, 3)),
            flux_clouds=None,
            phaseCent=[
                self.__dict__["centre"][0],
                self.__dict__["centre"][1],
            ],
            vOffset=self.__dict__["z_centre"],
        )

        # KinMS returns (x, y, v); autolens grids / ray-tracing expect (v, y, x).
        cube = cube.transpose(2, 1, 0)
        if instance.grid_3d is not None:
            target_shape = instance.grid_3d.shape_2d
            if cube.shape[1:] != target_shape:
                cube = np.stack(
                    [
                        resample_image_to_shape(channel, target_shape)
                        for channel in cube
                    ],
                    axis=0,
                )
        return cube

    # NOTE: ...
    def profile_cube_from_grid(
        self,
        grid_3d,
        z_step_kms: float,
        instance=None,
    ):

        if instance is None:
            raise NotImplementedError()

        return self.make_model(instance=instance)

    # NOTE: ...
    def profile_cube_from_masked_dataset(
        self,
        masked_dataset
    ):
        # log_xmin = np.log10(masked_dataset.pixel_scale / 5.0)
        # log_xmax = np.log10(masked_dataset.pixel_scale * 2.0 * masked_dataset.n_pixels)
        # x = np.logspace(
        #     log_xmin, log_xmax, 10000
        # )
        # return self.profile_cube_from_grid(
        #     x=x, instance=masked_dataset.instance,
        # )

        return self.profile_cube_from_grid(
            grid_3d=masked_dataset.grid_3d,
            z_step_kms=masked_dataset.z_step_kms,
            instance=masked_dataset.instance,
        )

# ============================================================================ #
# ============================================================================ #

class kinMSPixelized(Abstract):

    def __init__(
        self,
        centre=(0.0, 0.0),
        z_centre: float = 0.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
        vmax_black_hole: float = 0.0,
    ):
        super(kinMSPixelized, self).__init__()

        self.centre = centre
        self.z_centre = z_centre
        self.inclination = inclination
        self.phi = phi
        self.turnover_radius = turnover_radius
        self.maximum_velocity = maximum_velocity
        self.velocity_dispersion = velocity_dispersion
        self.vmax_black_hole = vmax_black_hole

    def make_model(self, instance):
        velprof = np.hypot(
            (2.0 * self.__dict__["maximum_velocity"] / np.pi)
            * np.arctan(instance.x / self.__dict__["turnover_radius"]),
            self.__dict__["vmax_black_hole"] / np.sqrt(instance.x),
        )

        # Phase-1 / truth SB maps are already on the sky (projected) source
        # plane. KinMS ``inClouds`` must stay at those sky positions: applying
        # ``inc`` again would re-project the morphology and brighten peaks by
        # ~1/cos(i). Subtract ``centre`` so kinematics sit on the free source
        # centre; ``phaseCent=[x, y]`` then places that centre in the cube.
        centre_x, centre_y = self.__dict__["centre"]
        in_clouds = np.asarray(instance.inClouds, dtype=float).copy()
        in_clouds[:, 0] -= float(centre_x)
        in_clouds[:, 1] -= float(centre_y)
        flux_clouds = np.asarray(instance.flux_clouds, dtype=float).copy()
        int_flux = instance.int_flux
        in_clouds, flux_clouds, int_flux = kinms_utils.apply_max_radius_to_clouds(
            in_clouds=in_clouds,
            flux_clouds=flux_clouds,
            int_flux=int_flux,
            max_radius=getattr(instance, "max_radius", None),
        )

        # Inclination / PA enter only through LOS velocities (KinMS skips
        # geometric projection when ``vLOS_clouds`` is provided).
        v_los = kinms_utils.sky_plane_vlos_kms(
            x_arcsec=in_clouds[:, 0],
            y_arcsec=in_clouds[:, 1],
            vel_rad_arcsec=instance.x,
            vel_prof_kms=velprof,
            inclination_deg=self.__dict__["inclination"],
            pos_ang_deg=self.__dict__["phi"],
            gas_sigma_kms=self.__dict__["velocity_dispersion"],
            seed=getattr(instance, "vlos_seed", 100),
        )

        cube = instance.obj.model_cube(
            inc=0.0,
            posAng=0.0,
            intFlux=int_flux,
            gasSigma=0.0,
            inClouds=in_clouds,
            flux_clouds=flux_clouds,
            vLOS_clouds=v_los,
            phaseCent=[
                float(centre_x),
                float(centre_y),
            ],
            vOffset=self.__dict__["z_centre"],
        )

        # KinMS returns (x, y, v); autolens grids / ray-tracing expect (v, y, x).
        cube = cube.transpose(2, 1, 0)
        if instance.grid_3d is not None:
            target_shape = instance.grid_3d.shape_2d
            if cube.shape[1:] != target_shape:
                cube = np.stack(
                    [
                        resample_image_to_shape(channel, target_shape)
                        for channel in cube
                    ],
                    axis=0,
                )
        return cube

    def profile_cube_from_grid(self, grid_3d, z_step_kms: float, instance=None):
        if instance is None:
            raise NotImplementedError()

        return self.make_model(instance=instance)

    def profile_cube_from_masked_dataset(self, masked_dataset):
        return self.profile_cube_from_grid(
            grid_3d=masked_dataset.grid_3d,
            z_step_kms=masked_dataset.z_step_kms,
            instance=masked_dataset.instance,
        )

# ============================================================================ #
# ============================================================================ #
