import numpy as np
import matplotlib.pylab as plt
# from getdist import plots, MCSamples, loadMCSamples
import corner
import sys
import pandas as pd

# sys.path.insert(0,'/Users/hstacey/packages/')
# from mplstyle import style
# style.make_science()

# aux = pandas.read_csv("samples.csv")
aux = np.loadtxt("samples.csv",skiprows=1,delimiter=",")

labels = pd.read_csv("samples.csv").columns.tolist()
print(labels)
# labels=["x","y","thetaE","ell0","ell1","gamma0","gamma1","reg"]

w = aux[:,-1]
x = aux[:,:-4]
# x[:,-1]=np.log10(x[:,-1])
# x[:,5]=2*x[:,5]+1

lens_pars=x[np.argmax(aux[:,-1]),:]
print(lens_pars)

fig=corner.corner(x, weights=w, plot_datapoints=True, plot_density=False, plot_contours=False, truths=lens_pars, bins=50, labels=labels[:-4])

fig.savefig("cornerplot.png",format="png")
plt.close(fig)
