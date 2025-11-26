import numpy as np
from ..utils.constants import mu0, pi
from ..utils.geometry import get_R, get_normal

def JxB_force(p, current, B):
    """
    Compute Lorentz force density: F = I (dl × B)
    Here dl = gradient of path node positions.
    """
    dl = np.gradient(p, axis=0)
    return current * np.cross(dl, B)


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