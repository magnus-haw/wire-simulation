import numpy as np
from scipy.special import ellipk, ellipe

from wireflux.physics.inductance import inductance
from wireflux.utils.constants import mu0, pi

def make_straight_wire(n=400, length=1.0, axis="z"):
    s = np.linspace(-length/2, length/2, n)
    if axis == "z":
        return np.column_stack((np.zeros_like(s), np.zeros_like(s), s))
    if axis == "x":
        return np.column_stack((s, np.zeros_like(s), np.zeros_like(s)))
    if axis == "y":
        return np.column_stack((np.zeros_like(s), s, np.zeros_like(s)))
    raise ValueError


def make_arc(n=1000, radius=1.0,theta_f=2*np.pi, z=0.0):
    theta = np.linspace(0, theta_f, n, endpoint=False)
    return np.column_stack((
        radius*np.cos(theta),
        radius*np.sin(theta),
        z*np.ones_like(theta)
    ))


def make_helix(n=1000,radius=1.0,pitchAngle=np.pi/4,curve_length=1.0):
    a = radius
    c = a*np.tan(pitchAngle)
    b = np.sqrt(a**2 + c**2)
    
    s = np.linspace(0, curve_length, n, endpoint=False)
    return np.column_stack((
        a*np.sin(s/b),
        a*np.cos(s/b),
        c*s/b
    ))