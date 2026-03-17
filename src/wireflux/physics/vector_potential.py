import numpy as np
import pyvista as pyv
from numpy.linalg import norm
from ..utils.constants import mu0, pi


def make_grid(boundaries,granularity=50,coordSys='cylindrical'):
    """
    Create a grid of points on which to compute values. Granularity controls the number of points in each dimension;
    meshing is not dynamic or asymmetric, for now. Coordinates can be generated in cylindrical, spherical, or 
    cartesian coordinates--however, the grid is always returned in cartesian coordinates. Distances will not be 
    square in non-cartesian grids.

    Parameters:
    -----------
    boundaries : ndarray
        Array of terminal boundaries of grid, as a list of tuples.
    granularit : int
        Number of points in each dimension of grid
    coordSys : string
        Type of coordinate system.
    """
    b1 = np.linspace(boundaries[0,0],boundaries[0,1],granularity)
    b2 = np.linspace(boundaries[1,0],boundaries[1,1],granularity)
    b3 = np.linspace(boundaries[2,0],boundaries[2,1],granularity)
    
    if (coordSys=='cylindrical' or coordSys=='cyl'):      # TODO: I am not sure if this is the smartest way to do this... I want the grid to be returned as a cartesian grid so that the difference methods are all easy, but I want grid points to conform to specific common useful boundary shapes... Need to think on this more after a good first draft is done. -JQM20260317
        q1 = b1*np.cos(b2)   # x = rho*cos(theta)
        q2 = b1*np.sin(b2)   # y = rho*sin(theta)
        q3 = b3              # z = z
        
    elif (coordSys=='spherical' or coordSys=='sph'):
        q1 = b1*np.cos(b2)*np.sin(b3)   # x = r*cos(theta)*sin(phi)
        q2 = b1*np.sin(b2)*np.sin(b3)   # y = r*sin(theta)*sin(phi)
        q3 = b1*np.cos(b3)              # z = r*cos(phi)
        
    elif (coordSys=='toroidal' or coordys=='tor'):          # NOTE: This is almost certainly overkill... -JQM20260317
        q1 = 1*np.sinh(b1)*np.cos(b3)/(np.cosh(b1)-np.cos(b2))   # x = a*sinh(tau)*cos(phi)/(cosh(tau)-cos(sigma))
        q1 = 1*np.sinh(b1)*np.sin(b3)/(np.cosh(b1)-np.cos(b2))   # x = a*sinh(tau)*sin(phi)/(cosh(tau)-cos(sigma))
        q1 = 1*np.sin(b2)/(np.cosh(b1)-np.cos(b2))               # x = a*sin(sigma)/(cosh(tau)-cos(sigma))
        
    elif (coordSys=='cartesian' or coordSys=='cart'):
        q1 = b1; q2 = b2; q3 = b3

    else:
        print("Unknown coordinate system requested. Try cyl, sph, or cart.")
        return np.zeros((granularity,granularity,granularity))      # TODO: Return grid of zeros or return error? -JQM20260317
    
    Q1,Q2,Q3 = np.meshgrid(q1, q2, q3, indexing='xy')

    grid = [Q1,Q2,Q3] 
    
    return grid


def compute_potential_from_wire(p, I, path, delta=2., beta=1.e5):
    """
    Segment-based integral of current density (normalized units), to find magnetic vector potential at point p
    from given path.

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
    A : ndarray, shape (3,)
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

    # Integral sum of currents (normalized kernel) for coulomb gauge
    dA = dl / (rmag[:,newaxis])
    A = np.sum(dA, axis=0)

    # Normalization & physical scaling
    A *= mu0 * I / (4*np.pi) / 2.0

    return A


def getAField_fromWires(p, wires, beta=0.):
    """
    Compute the magnetic vector potential from a single wire on a given grid of points, with specified boundaries
    and conditions thereof. Precision is primarily controlled by wire and grid density. The wire does not lay on 
    the grid points, necessarily.

    Parameters:
    -----------
    p : ndarray, shape (M, 3)
        Observation point.
    wires : list
        Objects with attributes:
            - p : ndarray, shape (N, 3)   (wire nodes)
            - I : float                  (current)
            - r : float                  (regularization length)

    Returns
    -------
    A : ndarray, shape (3)
        Magnetic field at the observation point (physical units).
    """

    M = len(wires)
    Acontrib = np.zeros((M, 3))

    for i in range(M):
        path  = wires[i].p
        I     = wires[i].I
        delta = 2*wires[i].r
        
        Acontrib[i,:] = compute_potential_from_wire(p, I, path, delta=delta) 

    A = np.sum(Acontrib,axis=1)   # NOTE: For now, I am not vectorizing this computation because I am not computing A along a path, rather computing A at many points on a 3D grid--I'm not sure if it's worth it to vectorize the computation for a large meshgrid, for memory purposes... -JQM20260317
    
    return A
    

def laplaceSolver(grid, boundaries, boundaryConditions='dirichlet'):
    """
    Solve laplace's equation on a given grid with given boundary conditions.

    Parameters:
    -----------
    grid : ndarray
        Grid on which to solve laplace's equation.
    boundaries : ndarray
        Array of terminal values of volume.
    boundaryConditions : string
        Type of boundary.
    """
    
    solv = 0 
    
    return solv


def find_scalarPotential(source, grid, boundaries, boundaryConditions='free'):
    """
    Compute the magnetic scalar potential on a given grid of points, from a source outside the volume of interest,
    given the specified boundaries and conditions thereof. Typically, these boundaries will be considered free
    boundaries as a scalar potential will typically be long-lived and 'permeate' leaky flux conserving boundaries.

    Parameters:
    -----------
    source : newWire
        Wire to compute scalar potential contribution from, outside the volume of interest.
    grid : ndarray
        Grid of points to compute magnetic scalar potential on.
    boundaries : ndarray
        Array of terminal values of volume to compute magnetic scalar potential within.
    boundaryConditions : string
        Type of boundary, typically 'free'.
    """

    scalarPotential = 0
    
    return scalarPotential


def impute_vectorPotential(wires, 
                           grid,
                           boundaries, 
                           boundaryPotential,
                           boundaryConditions='dirichlet',
                           precisionLevel='fast'):
    """
    Impute the vector potential from an arrangement of wires, given a boundary surface and the A values thereof.
    Includes controls for gauge and precision level. The grid size primarily controls the accuracy, but for a 
    'fast' computation the A contributions from wires within the volume are ignored to reduce computation time.
    All imputations from the given curent distribution are performed in coulomb gauge, where div(A)=0.

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
    A : ndarray, shape (M, 3)
        Magnetic vector potential at each grid point (normalized units).
    """

    A = 0

    return A