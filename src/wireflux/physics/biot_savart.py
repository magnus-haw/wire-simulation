### Utility functions
from numpy import zeros,gradient as grad, linalg
from numpy import newaxis,pi
import numpy as np

def _biot_savart(p, I, path, delta=.01):
    '''Given a point, a current and its path, calculates the magnetic field at that point
       This function uses normalized units:
        e.g. positions are normalized by a radial length scale r~(r'/L0)
             current is normalized to a characteristic value I~(I'/I0)
             Bfield is normalized to B~(B'/B0) where B0 = (mu0 I0)/(2 pi L0)
    '''
    dl = grad(path, axis=0) #centered differences
    r = path-p
    rmag = linalg.norm(r, axis=1)
    rmag[rmag<= delta] = 1e6

    B = sum(np.cross(r,dl) / (rmag**3.)[:,newaxis])
    B *= I/2.
    return B

def biot_savart(p, I, path, delta=2., beta=1.e5):
    """
    Segment-based Biot-Savart law (normalized units).

    Parameters
    ----------
    p : array_like, shape (3,)
        Observation point.
    I : float
        Current (normalized).
    path : array_like, shape (N, 3)
        Wire node positions.
    delta : float
        Regularization length scale (normalized).

    Returns
    -------
    B : ndarray, shape (3,)
        Magnetic field at point p (normalized units).
    """

    # Segment vectors
    dl = path[1:] - path[:-1]              # (N-1, 3)

    # Segment midpoints
    mid = 0.5 * (path[1:] + path[:-1])     # (N-1, 3)

    # Vectors from segment midpoints to observation point
    r = mid - p                            # (N-1, 3)
    rmag = linalg.norm(r, axis=1)

    # Regularization to avoid singularity
    mask = rmag > delta
    rmag = np.where(mask, rmag, np.inf)

    # Biot–Savart sum (normalized kernel)
    dB = np.cross(r,dl) / ((rmag**3.)[:,newaxis])
    B = np.sum(dB, axis=0)

    # Normalization (matches your previous convention)
    B *= (mu0 * I / 2.0 / pi)

    return B

def _getBField(path, wires):
    '''
    Given a path & a list of wires, returns B field at points along path
    '''
    n = len(path)
    B = zeros((n, 3))
    for p in range(n):
        for wire in wires:
            B[p,:] += biot_savart(path[p], wire.I, wire.p,delta=wire.r)
    return B

def getBField(path, wires, beta=0.):
    """
    Compute magnetic field at a set of observation points due to multiple wires,
    using a segment-based, vectorized Biot-Savart formulation.

    Parameters
    ----------
    path : ndarray, shape (M, 3)
        Observation points.
    wires : list
        Objects with attributes:
            - p : ndarray, shape (N, 3)   (wire nodes)
            - I : float                  (current)
            - r : float                  (regularization length)

    Returns
    -------
    B : ndarray, shape (M, 3)
        Magnetic field at each observation point (normalized units).
    """
    M = len(path)
    B = np.zeros((M, 3))

    for wire in wires:
        p = wire.p
        I = wire.I
        delta = 2*wire.r

        # Segment vectors and midpoints
        dl = p[1:] - p[:-1]                  # (Ns, 3)
        mid = 0.5 * (p[1:] + p[:-1])         # (Ns, 3)

        # Vector from segment midpoints to observation points
        # r[i, j] = mid[j] - path[i]
        r = mid[None, :, :] - path[:, None, :]   # (M, Ns, 3)

        rmag = linalg.norm(r, axis=2)                   # (M, Ns)

        # Regularization mask
        mask = rmag > delta
        rmag = np.where(mask, rmag, np.inf)

        # Biot–Savart kernel (vectorized)
        dB = np.cross(r,dl[None, :, :]) / (rmag[..., None]**3)

        # Sum over segments, scale by current
        B += (mu0 * I / 2.0 / pi) * np.sum(dB, axis=1)

    return B

