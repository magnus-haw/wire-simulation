import numpy as np
import pyvista as pv
from src.wireflux.models.wires import Wire

# ---------------------------------------------------------
# Create a circular loop (test geometry)
# ---------------------------------------------------------
N = 200
theta = np.linspace(0, 2*np.pi, N)

path = np.column_stack([
    np.cos(theta),
    np.sin(theta),
    np.zeros_like(theta)
])

path1 = np.column_stack([
    np.cos(theta),
    np.sin(theta),
    np.zeros_like(theta) + 3
])

v = np.zeros_like(path)
mass = np.ones((N, 1))
I = 1.0

# coil radius (tube thickness)
r = 0.05

# ---------------------------------------------------------
# Initialize Wire
# ---------------------------------------------------------
w1 = Wire(path, v, mass, I, r=r, is_fixed=False)
w2 = Wire(path1, v, mass, I, r=r, is_fixed=True)

# ---------------------------------------------------------
# Create and configure PyVista plotter
# ---------------------------------------------------------
plotter = pv.Plotter()
plotter.enable_lightkit()
plotter.set_background("white")

# Add wire to plotter
w1.show(plotter=plotter)
w2.show(plotter=plotter)

# Display
plotter.show()
