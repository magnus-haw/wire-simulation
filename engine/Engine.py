### Wire simulation, quasi-static approx

from numpy import array, zeros, arange, shape, apply_along_axis, abs, ones, matrix,dot
from numpy import diff, mgrid, tan,cos, sin, log, newaxis, linalg, cross, gradient as grad
from numpy import concatenate, arcsin,append, savetxt,mean,sqrt, linspace,arctan2,exp,cumsum
from scipy.interpolate import interp1d,splprep,splev
from functions import get_R, get_normal, get_disk,smooth,get_3D_curve_params
import pickle

def biot_savart(p, I, path, delta=.01):
    '''Given a point, a current and its path, calculates the magnetic field at that point
       This function uses normalized units:
        e.g. positions are normalized by a radial length scale r~(r'/L0)
             current is normalized to a characteristic value I~(I'/I0)
             Bfield is normalized to B~(B'/B0)
        These non-dimensional scalings are defined in Dimensions.py
    '''
    dl = grad(path, axis=0)
    r = path-p
    rmag = linalg.norm(r, axis=1)
    rmag[rmag<= delta] = 1e6

    B = sum(cross(r,dl) / (rmag**3.)[:,newaxis])
    B *= I/2.
    return B


def getBField(path, coil_paths, currents, delta=.01):
    '''
    Given the path of the loop, a list of coil paths (current paths), and
    a list of current magnitudes, returns B field at points along path
    '''
    n = len(path)
    B = zeros((n, 3))
    for p in range(n):
        for c in range(len(coil_paths)):
            B[p,:] += biot_savart(path[p], currents[c], coil_paths[c],delta=delta)
    return B


def JxB_force(path,I,B):
    '''
    Given a wire path, a current I and magnetic field B,
    calculates the JxB force at each point
    '''
    dl = grad(path, axis=0)
    return I*cross(dl,B)


def advance(path, dm, dt, allpaths, currents, velocity, delta=.012, extBfield = None):
    '''
    Given a uniform current-carrying "wire" specified with positions and velocities
    at many points along the "wire", updates the state due to magnetic forces after
    a small time-step has passed.
    
    Preconditions:
        For each i, velocity[i] corresponds to the segment at path[i]
        velocity[0] and velocity[-1] are zero (endpoints are fixed)
    '''
    
    # getBField uses biot_savart
    # TODO: Consider using an approximation to speed up the implementation of getBField?
    B = getBField(path, coil_paths, currents, delta=.01);
    if extBfield is not None:
        B+= extBfield(path)

    # Forces
    # TODO: add axial pressure?
    F = JxB_force(path,I,B)
    
    # Update path and velocity using acceleration from JxB force
    # TODO: Consider using a leapfrog/implicit/higher order numerical scheme here
    new_path = path + velocity * dt
    new_velocity = velocity + F / dm * dt

    # Fix first and final segments
    # TODO: continuity issues
    new_velocity[0,:]  = zeros(3)
    new_velocity[-1,:] = zeros(3)

    # TODO: Optimize interpolation and re-parametrize here?
    # TODO: variable mass density/ adaptive time steps
    # reinterpolate spline
    length_param = cumsum(linalg.norm(diff(new_path,axis=0),axis=1))
    length_param = append(0,length_param)
    f = interp1d(length_param, new_path, kind='cubic',axis=0)
    lp = linspace(0,length_param[-1],len(length_param))
    new_path = f(lp)
    
    return new_path, new_velocity
