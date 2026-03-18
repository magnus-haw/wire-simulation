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



def mutual_inductance(path1, path2, rwire=0.001, norm_mag=None, precision=15):
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
    precision : int
        Number of decimals to round values to
        
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
    L_mut = mu0 * Lsum / (4 * pi)

    if norm_mag is not None:
        L_mut /= norm_mag

    return L_mut



def self_inductance(path, rwire=0.001, norm_mag=None, precision=15):
    """
    Compute self inductance of any given wire
    using a segment-based Neumann formulation.
    From Majic 2024, "An integral for self-inductance of thin wires"

    Parameters
    ----------
    path : ndarray, shape (N, 3)
        Node coordinates of the wire path, endpts of segments.
    rwire : float
        Effective wire radius (regularization length).
    norm : float or None
        Optional normalization factor.
    precision : int
        Number of decimals to round values to

    Returns
    -------
    L : float
        Approximate self-inductance for a thin wire (SI units unless normalized).
    """

    # Segment vectors, segment unit vector, & end-to-end length
    dl = path[1:] - path[:-1]                                     # pt-to-pt vectors (N-1, 3)
    
    s        = np.linalg.norm(path - path[0,:],axis=1,keepdims=False)   # Progressive distance along path (N, 3)
    t        = np.zeros(path.shape)                                     
    t[:-1,:] = np.divide(dl, np.linalg.norm(dl,axis=1,keepdims=True))   # Unit vectors along path (N, 3)
    t[-1,:]  = t[-2,:]                                                  # Copy last unit vector into final position TODO: This should probably be replaced w/ BC normal vector, I feel -JQM20260316
    
    l  = s[-1]   # Last entry of s vector is total linear length of wire centroid

    # Summation approximation of shape inductance integral
    # TODO: I tried to vectorize this, for efficiency w/ long or multiple paths, but I feel this is probably a naive approach... -JQM20260317
    L_curve = np.nansum(np.round(np.divide(np.sum(           t[np.newaxis,:,:] *    t[:,np.newaxis,:], axis=2), 
                                           np.linalg.norm(path[np.newaxis,:,:] - path[:,np.newaxis,:], axis=2)), precision) - 
                        np.round(np.divide(1,np.abs(np.subtract.outer(s,s))), precision) ) 
                        # NOTE: I decided to just leave runtime warnings on, though every np.inf is subtracted by another np.inf--the nansum should always iron out to 0 in those positions. This is part of what makes the actual analytical computation feasible. -JQM2060317
    
    L_parr = 2*(l*np.log((l + np.sqrt(l**2 + rwire**2))/rwire)-np.sqrt(l**2+rwire**2)+l/4+rwire)   # L of long, small-radius circular wire w/ uniform J
    
    L_sum = L_curve + L_parr   # Approximate self-inductance is sum of 'shape inductance' integral & inductance of equivalent length straight wire (Majic, 2024)

    # Physical scaling
    L_self = mu0 * L_sum # / (4 * pi)   # TODO: Not sure if we need to divide by 4pi here... -JQM20260316

    if norm_mag is not None:
        L_self /= norm_mag

    return L_self



def inductance(path1, path2=None, rwire=0.001, norm_mag=None, part='mutual', precision=15):
    """
    Simple wrapper function for returning either mutual or self inductance of one or two paths.

    Parameters
    ----------
    path1 : ndarray, shape (N, 3)
        Node coordinates of the wire path, endpts of segments.
    path2 : ndarray, shape (N, 3)
        Node coordinates of second wire path, endpts of segments, for use with mutual inductance.
    rwire : float
        Effective wire radius (regularization length).
    norm : float or None
        Optional normalization factor.
    part : string
        String determining which part of inductance (self or mutual) is to be returned
    precision : int
        Number of decimals to round values to
        
    Returns
    -------
    L : float
        Approximate self-inductance for a thin wire (SI units unless normalized).
    """
    L = 0
    
    if (part=='mutual' or part=='Mutual' or part=='m' or part=='M'):
        if path2 is not None: L = mutual_inductance(path1,path2,rwire=rwire,norm_mag=norm_mag)
        else: print('Second path needed for mutual inductance calculation')
    
    elif (part=='self' or part=='Self' or part=='s' or part=='S'):
        L = self_inductance(path1,rwire=rwire,norm_mag=norm_mag)
    
    else:
        print('Unknown Inductance type requested')

    return L