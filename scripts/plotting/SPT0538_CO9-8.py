import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# import galpak
from spectral_cube import SpectralCube
from astropy.io import fits

pc_to_m=3.085e16
Msol_to_kg=2e30

def vel_kep_mass(r,M_sol,pc_arcsec):
    return 1e-3 * np.sqrt(6.67408e-11 * M_sol * Msol_to_kg / (r*pc_arcsec*pc_to_m))

n_pixels = 1024
pixel_scale = 1/n_pixels

scalerad = 0.234 # effective radius arcseconds
rt = 0.05
vmax = 320

veldisp = 35

pos = -11.932+90+180 # degrees
inc = 64.686 # degrees

xsize = 1 # arcsec
ysize = 1 # arcsec
dv = 30 # km/s/channel

type="KinMS"

if type=="Galpak":

    model = galpak.DiskModel(
        flux_profile='exponential',
        thickness_profile="gaussian",
        rotation_curve='isothermal',
        dispersion_profile="thick"
    )

    """
    x (pix)
    y (pix)
    z (pix)
    flux
    radius aka. r½ (pix) half-light radius in pixel
    inclination (deg)
    pa (deg) position angle from y-axis, anti-clockwise.
    turnover_radius aka. rv (pix) turnover radius for arctan, exp, velocity profile [is ignored for mass profile]
    maximum_velocity aka. Vmax (km/s) de-projected V_max [forced to be positive, with 180deg added to PA]
    velocity_dispersion aka. s0 (km/s)
    """

    galaxy = galpak.GalaxyParameters.from_ndarray(
        a=np.array([50,50,34/2,100,scalerad,inc,pos,rt,vmax,veldisp])
    )
    cube, _, _, _ = model._create_cube(
        galaxy=galaxy,
        shape=(34,100,100),
        z_step_kms=dv,
        zo=0
    )

    print(cube.data.shape)

    newcube=np.moveaxis(cube.data, 0, -1)
    newcube=np.moveaxis(newcube, 1, 0)
    print(newcube.shape)

    fits.writeto("test_cube.fits",newcube,overwrite=True)
                        
    # hdul = fits.open('test_cube.fits')[0]
    # print(hdul.shape)
    # hdul.header["NAXIS"] = 3
    # hdul.header["CDELT1"] = 1.0e-4  #degrees
    # hdul.header["CDELT2"] = 1.0e-4  #degrees
    # hdul.header["CDELT3"] = 30 #km/s
    # hdul.header["CTYPE1"] = 'RA---CAR'
    # hdul.header["CTYPE2"] = 'DEC--CAR'
    # hdul.header["CTYPE3"] = 'VRAD'
    # hdul.header["CUNIT1"] = 'deg'
    # hdul.header["CUNIT2"] = 'deg'
    # hdul.header["CUNIT3"] = 'km/s'
    # hdul.header["CRVAL1"] = 0
    # hdul.header["CRVAL2"] = 0
    # hdul.header["CRVAL3"] = 0

    # fits.writeto("test_cube.fits",data=hdul.data,header=hdul.header,overwrite=True)

    # # vrange = np.arange(-1*34/2,34/2,34)*30

    # cube = SpectralCube.read("test_cube.fits") 
    # moment_0 = cube.moment(order=0)  
    # moment_1 = cube.moment(order=1)  
    # moment_2 = cube.moment(order=2) 

    # fig,[ax1,ax2,ax3] = plt.subplots(1,3)

    # ax1.imshow(cube[:,:,0],origin="lower",cmap="RdBu_r")

    # ax2.imshow(cube[:,:,10],origin="lower",cmap="RdBu_r")

    # ax3.imshow(cube[:,:,20],origin="lower",cmap="RdBu_r")

    # fig.savefig("test_plot.pdf")

else:

    # from kinms import KinMS
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from kinms.utils.sauron_colormap import sauron

    # from kinms.utils.KinMS_figures import KinMS_plotter

    # radius = np.arange(0, 0.3, 0.005) # radius vector in arcseconds
    # sbprof = np.exp(-radius / scalerad)
    # velprof = 2.0 * vmax / np.pi * np.arctan(radius / rt) #+ vel_kep_mass(radius,1e8,8000)

    # # cellsize = 0.005 # arcsec/pixel
    # dv = 30 # km/s/channel
    # beamsize = [0.01, 0.01, 0] # arcsec, arcsec, degrees
    # vsize = 30*34

    # kin = KinMS(xsize, ysize, vsize, pixel_scale, dv, beamSize = beamsize, verbose = True)  

    # cube = kin.model_cube(inc = inc, sbProf = sbprof, gasSigma = veldisp, #phaseCent=[-0.071,0.097],
    #             sbRad = radius, velRad = radius, velProf = velprof, posAng = pos, toplot=True)  

    # fits.writeto("test_cube.fits",cube,overwrite=True)

    hdul = fits.open('test_cube.fits')[0]
    print(hdul.shape)
    hdul.header["NAXIS"] = 3
    hdul.header["CDELT1"] = 1.0e-4  #degrees
    hdul.header["CDELT2"] = 1.0e-4  #degrees
    hdul.header["CDELT3"] = 30 #km/s
    hdul.header["CTYPE1"] = 'RA---CAR'
    hdul.header["CTYPE2"] = 'DEC--CAR'
    hdul.header["CTYPE3"] = 'VRAD'
    hdul.header["CUNIT1"] = 'deg'
    hdul.header["CUNIT2"] = 'deg'
    hdul.header["CUNIT3"] = 'km/s'
    hdul.header["CRVAL1"] = 0
    hdul.header["CRVAL2"] = 0
    hdul.header["CRVAL3"] = -39 * 34/2

    fits.writeto("test_cube.fits",data=hdul.data,header=hdul.header,overwrite=True)

    cube = SpectralCube.read("test_cube.fits") 
    moment_0 = cube.moment(order=0, axis=2)
    moment_1 = cube.moment(order=1, axis=2)  
    moment_2 = cube.moment(order=2, axis=2) 

    fig,[ax1,ax2,ax3] = plt.subplots(1,3)

    ax1.imshow(moment_0.value,origin="lower",cmap="RdBu_r")

    ax2.imshow(moment_1.value,origin="lower",cmap="RdBu_r")

    ax3.imshow(moment_2.value,origin="lower",cmap="RdBu_r")

    fig.savefig("test_plot.pdf")

    fits.writeto("test_cube_mom1.fits",moment_1.value,overwrite=True)