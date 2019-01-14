### CroFT simulation, quasi-static approx

from Constants import mu0, pi
from numpy import array, zeros, arange, shape, abs, ones, matrix
from numpy import diff, mgrid, cos, sin, log, newaxis, linalg, cross,arccos
from numpy import concatenate, arcsin
import mayavi.mlab as mlab
import time

blue  = (0.34765625,0.5625,0.84375)
copper= (0.84765625,0.5625,0.34375)

def center_electrode():
    radius = 0.2 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) 
    z = 0*theta + .0254/8.
    mlab.mesh(x, y, z, color=blue)
    mlab.mesh(x, y, z*0, color=blue)

    dz = .0254/8.
    [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) 
    mlab.mesh(x, y, z, color=blue)

    inners,outers = get_jet_nozzles()
    for p in inners:
        radius = .323*.0254 #meters
        dr, dtheta = radius/10.0, pi/10.0
        [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
        x = r*cos(theta)-p[0]
        y = r*sin(theta)-p[1]
        z = 0*theta + .0254/8.+.0001
        mlab.mesh(x, y, z, color=(0,0,0))

    return array([0,9.75*.0254])

def annulus_electrode():
    rin,rout = 0.21,0.5 #meters
    dr, dtheta = rin/10.0, pi/10.0

    ### top and bottom surfaces
    [r,theta] = mgrid[rin:rout+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)
    z = 0*theta + .0254/8.
    mlab.mesh(x, y, z, color=copper)
    mlab.mesh(x, y, z*0, color=copper)

    ### edges
    dz = .0254/8.
    for radius in [rin,rout]:
        [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
        x = radius*cos(theta)
        y = radius*sin(theta)
        mlab.mesh(x, y, z, color=copper)

    inners,outers = get_jet_nozzles()
    ### Gas ports
    for p in outers:
        radius = .323*.0254 #meters
        dr, dtheta = radius/10.0, pi/10.0
        [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
        x = r*cos(theta) - p[0]
        y = r*sin(theta) - p[1]
        z = 0*theta + .0254/8.+.0001
        mlab.mesh(x, y, z, color=(0,0,0))

    return array([0,-9.75*.0254])

def get_jet_nozzles():
    ro = 0.355 #meters
    ri = 0.10
    dphi = 2*pi/8.
    outer =[]
    inner =[]
    for phi in arange(0,2*pi,dphi):
        x,y = cos(phi),sin(phi)
        outer.append([x*ro,y*ro])
        inner.append([x*ri,y*ri])
    return inner,outer

def get_stuff_coil(dx=0, dy=0, r=1.623*.0254, d=.0254, nturns=4.):
    theta = arange(0,2*pi*nturns,pi/15.)
    path = array([r*cos(theta)+dx,r*sin(theta)+dy,-d*theta/(2*pi*nturns)])
    return path

                
