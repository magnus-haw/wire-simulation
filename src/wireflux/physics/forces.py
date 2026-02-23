import numpy as np
from ..utils.constants import mu0, pi
from ..utils.geometry import get_R, get_normal

import numpy as np

def JxB_force(p, current, B):
    """
    Compute Lorentz force density F on a wire with path p and
    magnetic field B using a segment-based formulation.

    Force is computed per segment and then distributed to the adjacent nodes:
        dF_seg = I * (dl_seg x B_mid)
    where dl_seg = p[i+1] - p[i],
          B_mid = average of B at the segment endpoints.

    The resulting force on each node is the average of neighboring segment forces.

    Parameters
    ----------
    p : np.ndarray, shape (N, 3)
        Positions of the wire nodes.
    current : float
        Current magnitude along the wire.
    B : np.ndarray, shape (N, 3)
        Magnetic field evaluated at the node positions.

    Returns
    -------
    F_node : np.ndarray, shape (N, 3)
        Force per node.
    
    # NOTE:
    # Force is computed per segment and distributed to nodes.
    # This formulation is invariant under reparameterization
    # of the wire path (up to discretization error).
    """
    # Number of nodes
    N = len(p)

    if N < 2:
        return np.zeros_like(p)

    # Compute segment vectors
    dl_seg = p[1:] - p[:-1]  # shape (N-1, 3)

    # Compute midpoint B for each segment
    B_mid = 0.5 * (B[:-1] + B[1:])  # shape (N-1, 3)

    # Compute segment-level force
    F_seg = current * np.cross(dl_seg, B_mid)  # shape (N-1, 3)

    # Distribute segment forces to nodes
    F_node = np.zeros_like(p)

    # Add half of each segment's force to each endpoint
    F_node[:-1] += 0.5 * F_seg
    F_node[1:]  += 0.5 * F_seg

    return F_node

def pressure_repulsion(p_i, p_j, k_rep, dr, power=2):
    """
    Compute repulsive force between two wires.
    Returns (F_on_i, F_on_j).
    """

    cutoff = dr

    F_i = np.zeros_like(p_i)
    F_j = np.zeros_like(p_j)

    diff = p_i[:, None, :] - p_j[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    mask = (dist < cutoff) & (dist > 1e-12)

    if not np.any(mask):
        return F_i, F_j

    r_hat = np.zeros_like(diff)
    r_hat[mask] = diff[mask] / dist[mask][:, None]

    strength = np.zeros_like(dist)
    strength[mask] = k_rep * (1 - dist[mask] / cutoff) ** power

    F_pair = strength[:, :, None] * r_hat

    # Sum contributions
    F_i += np.sum(F_pair, axis=1)
    F_j -= np.sum(F_pair, axis=0)  # equal & opposite

    return F_i, F_j


def tension_force(wire):
    '''
    Calculates tension force from 3D curve properties
    '''
    T,CumLen,dl,N,R,tck,s = wire.get_3D_curve_params()
    vol = np.pi*wire.r*wire.r*dl
    Lsq = CumLen[-1]**2
    ft = vol*(wire.Bp*wire.Bp/R)*((Lsq - wire.L_init**2)/Lsq)*N.T
    return ft.T

def tension_force1(R,N,L,L0,Phi,a,dl):
    ## tension force
    ft = dl*(Phi*Phi/(pi*a*a*2*mu0*R))*((L*L-L0*L0)/L/L)*N.T
    return ft.T

def tension_force0(path,B):
    '''
    Given a path, shape(n,3), and a B-field magnitude,
    calculates magnetic tension force from path curvature
    '''
    ## Radius of curvature
    R = get_R(path.T)
    rhat= get_normal(path.T)
    
    ## tension force
    ft = B*B*(1/R)*rhat.T/mu0
    return ft.T

