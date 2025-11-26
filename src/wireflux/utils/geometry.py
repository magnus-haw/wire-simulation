import numpy as np
from numpy import sqrt, cumsum, array
from scipy.interpolate import splprep, splev
from numpy import gradient as grad

### Use cubic splines to calculate path derivatives
def get_3D_curve_params(x,y,z):
    tck,s = splprep([x,y,z],s=0)
    ds = s[1]-s[0]
    dx,dy,dz = splev(s,tck,der=1)
    dr = sqrt(dx*dx + dy*dy + dz*dz)
    dl = dr*ds
    L = cumsum(dl)
    T = array([dx/dr,dy/dr,dz/dr]).T

    p,u = splprep([dx/dr,dy/dr,dz/dr],u=L,s=0)
    dTx,dTy,dTz = splev(u,p,der=1)
    kurv = np.sqrt(dTx*dTx + dTy*dTy + dTz*dTz)
    kurv[kurv==0] = 1e-14
    N = array([dTx/kurv,dTy/kurv,dTz/kurv]).T
    R = 1./kurv

    # return tangent, length, dlength, normal vector, radius of curvature
    return T,L,dl,N,R


def get_R(r):
    '''Solve for curvature in 3d (assume x(s), y(s), z(s)).
       endpoints and penultimate points will give
       incorrect values due to boundary effects
    '''
    xp = grad(r[0])
    xpp= grad(xp)

    yp = grad(r[1])
    ypp= grad(yp)

    zp = grad(r[2])
    zpp= grad(zp)

    numer = ( (zpp*yp - ypp*zp)**2. + (xpp*zp - zpp*xp)**2. + (ypp*xp - xpp*yp)**2. )**.5
    denom = (xp**2. + yp**2. + zp**2.)**1.5
    kappa = numer/denom

    return 1./kappa
    
def get_normal(r):
    '''
    Solve for normal unit vector of 3D curve, r=(x,y,z)
    '''
    xp = grad(r[0])
    yp = grad(r[1])
    zp = grad(r[2])
    
    v = (xp**2 + yp**2 + zp**2)**.5
    
    tx = xp/v
    ty = yp/v
    tz = zp/v
    
    nx = grad(tx)
    ny = grad(ty)
    nz = grad(tz)
    norm = (nx**2 + ny**2 + nz**2)**.5
    N = array([nx,ny,nz])/norm

    return N.T

# -------------------------------------------------------------------
# BASIC FINITE-DIFFERENCE GEOMETRY (Used in NewWire dynamic remeshing)
# -------------------------------------------------------------------

def arclength(path):
    """
    Compute segment lengths and cumulative arclength via FD differences.
    """
    diffs = np.roll(path, -1, axis=0) - path
    ds = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    return ds, s


def tangent_fd(path):
    """
    Centered-difference tangent, normalized.
    """
    fwd = np.roll(path, -1, axis=0)
    bwd = np.roll(path, 1, axis=0)
    T = fwd - bwd
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return T / n


def curvature_fd(path):
    """
    Discrete curvature using dT/ds and FD geometry.
    """
    ds, _ = arclength(path)
    T = tangent_fd(path)

    dT = np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0)
    dT_norm = np.linalg.norm(dT, axis=1)

    return dT_norm / np.maximum(ds, 1e-12)