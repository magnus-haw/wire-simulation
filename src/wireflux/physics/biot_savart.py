### Utility functions
from numpy import zeros,gradient as grad,linalg
from numpy import newaxis,pi
import numpy as np

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

def getBField(path, wires):
    '''
    Given a path & a list of wires, returns B field at points along path
    '''
    n = len(path)
    B = zeros((n, 3))
    for p in range(n):
        for wire in wires:
            B[p,:] += biot_savart(path[p], wire.I, wire.p,delta=wire.r)
    return B


