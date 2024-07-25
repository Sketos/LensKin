import os, sys
import numpy as np

# NOTE:
import autofit as af
import autolens as al

from autoastro import dimensions as dim

# ---------------------------------------------------------------------------- #
# NOTE import GALPACK3D
path = os.getenv("HOME") + "/Desktop/GitHub/UVgalpak3D"
sys.path.append(path)
import galpak
# ---------------------------------------------------------------------------- #

class Abstract(al.lp.LightProfile):
    def __init__(self):
        pass

# ============================================================================ #
# ============================================================================ #

class Kinematical(Abstract):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        z_centre: float = 0.0,
        intensity: float = 0.1,
        effective_radius: float = 1.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
    ):
        super(
            Kinematical, self
        ).__init__()

        self.centre = centre
        self.z_centre = z_centre
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.inclination = inclination
        self.phi = phi
        self.turnover_radius = turnover_radius
        self.maximum_velocity = maximum_velocity
        self.velocity_dispersion = velocity_dispersion

    def convert_centre_from_arcsec_to_pixels(
        self,
        value,
        pixel_scale,
        n_pixels,
    ):
        return value / pixel_scale + n_pixels / 2.0


    def convert_radius_from_arcsec_to_pixels(
        self,
        value,
        pixel_scale,
    ):
        return value / pixel_scale


    # NOTE: grid should be replaced with grid_3d (working on it)
    # NOTE: Maybe not the best way to do this ... (It is very fast the way it is)
    def convert_parameters(self, grid):
        #start = time.time()

        # NOTE:
        #galpak = (x, y)
        #autolens_centre = (y, x)

        names = []

        # NOTE: "converted_parameters" can also be a numpy array.
        converted_parameters = []
        for i, (name, value) in enumerate(
            self.__dict__.items()
        ):
            if name not in ["id", "_assertions", "cls"]:

                if name == "centre":
                    names.append("centre_0")
                    names.append("centre_1")
                else:
                    names.append(name)

                if name == "centre":
                    for (i, sign) in zip([1, 0], [1.0, -1.0]):
                        #print(value, sign)
                        converted_parameters.append(
                            self.convert_centre_from_arcsec_to_pixels(
                                value=sign * value[i],
                                pixel_scale=grid.pixel_scale,
                                n_pixels=grid.shape_2d[i]
                            )
                        )
                    #exit()
                elif name.endswith("radius"):
                    converted_parameters.append(
                        self.convert_radius_from_arcsec_to_pixels(
                            value=value,
                            pixel_scale=grid.pixel_scale
                        )
                    )
                else:
                    converted_parameters.append(value)
        # end = time.time()
        # print(
        #     "It took t={} to convert parameters".format(end - start)
        # )

        print("parameters (converted)", converted_parameters)

        return converted_parameters


    # NOTE: ...
    def profile_cube_from_grid(
        self,
        grid_3d,
        z_step_kms
    ):

        # NOTE: ...
        model = galpak.DiskModel(
            flux_profile='exponential',
            thickness_profile="gaussian",
            rotation_curve='isothermal',
            dispersion_profile="thick"
        )
        galaxy = galpak.GalaxyParameters.from_ndarray(
            a=self.convert_parameters(grid=grid_3d.grid_2d)
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

        return self.profile_cube_from_grid(
            grid_3d=masked_dataset.grid_3d,
            z_step_kms=masked_dataset.z_step_kms,
        )


# ============================================================================ #
# ============================================================================ #

class kinMS(Abstract):

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
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


    def make_model(
        self,
        instance,
        x: np.ndarray
    ):

        # NOTE:
        sbprof = np.exp(-x / self.__dict__["effective_radius"])

        # NOTE:
        #velprof = (2.0 * self.__dict__["maximum_velocity"] / np.pi) * np.arctan(x / self.__dict__["turnover_radius"]) + self.__dict__["vmax_black_hole"] / np.sqrt(x)
        velprof = np.hypot(
            (2.0 * self.__dict__["maximum_velocity"] / np.pi) * np.arctan(x / self.__dict__["turnover_radius"]),
            self.__dict__["vmax_black_hole"] / np.sqrt(x)
        )

        # NOTE:
        cube = instance.model_cube(
            inc=self.__dict__["inclination"],
            posAng=self.__dict__["phi"],
            intFlux=self.__dict__["intensity"],
            gasSigma=self.__dict__["velocity_dispersion"],
            #diskThick=0.1,
            sbProf=sbprof,
            velProf=velprof,
            sbRad=x,
            velRad=x,
            phaseCent=[
                -self.__dict__["centre"][0],
                self.__dict__["centre"][1]
            ],
            vOffset=self.__dict__["z_centre"],
            #toplot=True,
        )

        return cube.transpose(2, 0, 1)

    # NOTE: ...
    def profile_cube_from_grid(
        self,
        x,
        instance, # NOTE: This is unique to kinMS
    ):

        return self.make_model(instance, x)

    # NOTE: ...
    def profile_cube_from_masked_dataset(
        self,
        masked_dataset
    ):
        log_xmin = np.log10(masked_dataset.pixel_scale / 5.0)
        log_xmax = np.log10(masked_dataset.pixel_scale * 2.0 * masked_dataset.n_pixels)
        x = np.logspace(
            log_xmin, log_xmax, 10000
        )
        return self.profile_cube_from_grid(
            x=x, instance=masked_dataset.instance,
        )

# ============================================================================ #
# ============================================================================ #


"""
class Kinematical(al.lp.Kinematical):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        z_centre: float = 0.0,
        intensity: float = 0.1,
        effective_radius: float = 1.0,
        inclination: float = 0.0,
        phi: float = 50.0,
        turnover_radius: float = 0.0,
        maximum_velocity: float = 200.0,
        velocity_dispersion: float = 50.0,
    ):
        super(Kinematical, self).__init__(
            centre=centre,
            z_centre=z_centre,
            intensity=intensity,
            effective_radius=effective_radius,
            inclination=inclination,
            phi=phi,
            turnover_radius=turnover_radius,
            maximum_velocity=maximum_velocity,
            velocity_dispersion=velocity_dispersion,
        )
"""
"""
# NOTE: added
class kinMS(al.lp.kinMS):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
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
        super(kinMS, self).__init__(
            centre=centre,
            z_centre=z_centre,
            intensity=intensity,
            effective_radius=effective_radius,
            inclination=inclination,
            phi=phi,
            turnover_radius=turnover_radius,
            maximum_velocity=maximum_velocity,
            velocity_dispersion=velocity_dispersion,
            vmax_black_hole=vmax_black_hole,
        )
"""