### Utility functions
from numpy import array,zeros,gradient as grad,linalg,cross
from numpy import newaxis,shape,sqrt,pi
from scipy.interpolate import splprep,splev,splrep,griddata
import numpy as np
mu0=4e-7*pi

def biot_savart(p, I, path, delta=.01):
    '''Given a point, a current and its path, calculates the magnetic field at that point
       This function uses normalized units:
        e.g. positions are normalized by a radial length scale r~(r'/L0)
             current is normalized to a characteristic value I~(I'/I0)
             Bfield is normalized to B~(B'/B0)
    '''
    dl = grad(path, axis=0) #centered differences
    r = path-p
    rmag = linalg.norm(r, axis=1)
    rmag[rmag<= delta] = 1e6

    B = sum(np.cross(r,dl) / (rmag**3.)[:,newaxis])
    B *= I/2.
    return B