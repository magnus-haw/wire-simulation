import numpy as np
from numpy.linalg import norm
from ..utils.constants import mu0, pi


def _inductance(path1,path2,rwire=0.001,norm=None):
    '''
    Given two wire paths, calculates the mutual inductance in SI units
    taken from: Advanced Electromagnetics. 2016;5(1)
    DOI 10.7716/aem.v5i1.331 
    '''
    dl1 = np.gradient(path1, axis=0) #centered differences
    dl2 = np.gradient(path2, axis=0) #centered differences
    L = 0
    for i in range(0,len(path1)):        
        for j in range(0,len(path2)):
            dd = np.sqrt( ((path1[i] - path2[j])**2).sum() )
            if dd > rwire/2.:
                L+=np.dot(dl1[i],dl2[j])/dd
    if norm is None:
        return mu0*L/(4*pi)
    else:
        return mu0*L/(4*pi)/norm


def inductance(path1, path2, rwire=0.001, norm_mag=None):
    """
    Compute mutual inductance between two wires
    using a segment-based Neumann formulation.

    Parameters
    ----------
    path1, path2 : ndarray, shape (N, 3)
        Node coordinates of the two wire paths.
    rwire : float
        Effective wire radius (regularization length).
    norm : float or None
        Optional normalization factor.

    Returns
    -------
    L : float
        Mutual inductance (SI units unless normalized).
    """

    # Segment vectors
    dl1 = path1[1:] - path1[:-1]      # (N1-1, 3)
    dl2 = path2[1:] - path2[:-1]      # (N2-1, 3)

    # Segment midpoints
    mid1 = 0.5 * (path1[1:] + path1[:-1])   # (N1-1, 3)
    mid2 = 0.5 * (path2[1:] + path2[:-1])   # (N2-1, 3)

    # Vectorized distance matrix
    r = mid1[:, None, :] - mid2[None, :, :]   # (N1-1, N2-1, 3)
    rmag = norm(r, axis=2)

    # Regularization: avoid singular self / near interactions
    rmag = np.maximum(rmag, rwire)

    # Dot product of segment vectors
    dot = np.einsum("ik,jk->ij", dl1, dl2)

    # Neumann sum
    Lsum = np.sum(dot / rmag)

    # Physical scaling
    L = mu0 * Lsum / (4 * pi)

    if norm_mag is not None:
        L /= norm_mag

    return L