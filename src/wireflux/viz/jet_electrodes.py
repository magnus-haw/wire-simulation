### jet experiment electrodes & coils
import pyvista as pv
import numpy as np
from wireflux.utils.constants import pi
from wireflux.viz.load_stl import add_stl_to_plotter
from numpy import array, arange

blue  = (0.34765625,0.5625,0.84375)
copper= (0.84765625,0.5625,0.34375)

def jet_electrodes(plotter=None):
    if plotter is None:
        plotter = pv.Plotter()
        plotter.set_background("white")
    plotter, mesh1 = add_stl_to_plotter(plotter, "/Users/mhaw/Desktop/ARCTRON/software/wire-simulation/src/wireflux/viz/jet_annular_electrode.stl", 
                                        color=(0.2, 0.4, 0.9))
    plotter, mesh2 = add_stl_to_plotter(plotter, "/Users/mhaw/Desktop/ARCTRON/software/wire-simulation/src/wireflux/viz/jet_inner_elec.stl", 
                                        color=(0.85, 0.56, 0.34))

def get_jet_nozzles():
    ro = 0.355 #meters
    ri = 0.10
    dphi = 2*pi/8.
    outer =[]
    inner =[]
    for phi in arange(0,2*pi,dphi):
        x,y = np.cos(phi),np.sin(phi)
        outer.append([x*ro,y*ro])
        inner.append([x*ri,y*ri])
    return inner,outer

def get_stuff_coil(dx=0, dy=0, r=1.623*.0254, d=.0254, nturns=4.):
    theta = arange(0,2*pi*nturns,pi/15.)
    path = array([r*np.cos(theta)+dx,r*np.sin(theta)+dy,-d*theta/(2*pi*nturns)])
    return path


if __name__ == "__main__":
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter, mesh1 = add_stl_to_plotter(plotter, "/Users/mhaw/Desktop/ARCTRON/software/wire-simulation/src/wireflux/viz/jet_annular_electrode.stl", 
                                        color=(0.2, 0.4, 0.9))
    plotter, mesh2 = add_stl_to_plotter(plotter, "/Users/mhaw/Desktop/ARCTRON/software/wire-simulation/src/wireflux/viz/jet_inner_elec.stl", 
                                        color=(0.85, 0.56, 0.34))

    plotter.show()

