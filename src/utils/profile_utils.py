import numpy as np

def axis_ratio_and_phi_from(elliptical_comps):

    angle = np.arctan2(elliptical_comps[0], elliptical_comps[1]) / 2
    angle *= 180.0 / np.pi
    fac = np.sqrt(elliptical_comps[1] ** 2 + elliptical_comps[0] ** 2)
    if fac > 0.999:
        fac = 0.999  # avoid unphysical solution
    # if fac > 1: print('unphysical e1,e2')
    axis_ratio = (1 - fac) / (1 + fac)

    return axis_ratio, angle

def shear_magnitude_and_phi_from(elliptical_comps):
    
    angle = np.arctan2(elliptical_comps[0], elliptical_comps[1]) / 2 * 180.0 / np.pi
    magnitude = np.sqrt(elliptical_comps[1] ** 2 + elliptical_comps[0] ** 2)
    if angle < 0:
        return magnitude, angle + 180.0
    else:
        return magnitude, angle
