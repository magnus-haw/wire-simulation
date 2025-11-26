import numpy as np
import pyvista as pv
from src.wireflux.models.wires import Wire

# spiral example
N = 300
t = np.linspace(0, 4*np.pi, N)

path = np.column_stack([np.cos(t), np.sin(t), 0.2*t])
mass = np.ones((N,1))
v = 0.05*np.column_stack([-np.sin(t), np.cos(t), 0.2*np.ones_like(t)])
forces = np.column_stack([np.sin(2*t), 0.5*np.cos(3*t), 0.1*np.sin(4*t)])

w = Wire(path, v, mass, I=1.0, r=0.03)

plotter = pv.Plotter()
plotter.set_background("black")

# Add the wire
w.show(plotter=plotter, forces=forces, velocity=True)

plotter.show()
