import numpy as np
import pyvista as pv
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from classes.Wires import Wire
from classes.State import State
from classes.Utility import biot_savart,getBField, get_R, get_normal, get_rect_grid
from classes.Engine import MultiWireEngine

import matplotlib.pyplot as plt

### Dimensional scales
r0 = 0.0762 #m
L0 = .0254*1.5 #m
I0 = 25. #Amps
nden0 = 1e21 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s
m0 = rho0*pi*r0*r0*L0
nturns = 59
n = 1310

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
L = L0/r0
dL = .456/r0
dr = 1.
dt = .02
I = 1.
rho = 1.
dm = pi*dr

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,2*pi*nturns,n)
path1 = np.array([dr*np.cos(phi),dr*np.sin(phi),-L/2. + phi*L/(2*pi*nturns)]).T
path2 = np.array([dr*np.cos(phi),dr*np.sin(phi),dL + -L/2. + phi*L/(2*pi*nturns)]).T

### Initialize mass
mass = np.ones((n,1))*dm
### Create wire 
wr = Wire(path1,path1*0,mass,I,r=.03,Bp=0, is_fixed=True)
wr2 = Wire(path2,path2*0,mass,I,r=.03,Bp=0, is_fixed=True)
############### Create intial state ##############
st = State('single_loop_test',load=0)
st.items.append(wr);st.items.append(wr2)
############## Run simulation engine #############
sim = MultiWireEngine(st,dt)


probes = np.array([[0,0,0],[0,0,1],[0,0,2]])
B_vectors = []
for i in range(0,len(probes)):
    B = biot_savart(probes[i], I, wr.p, delta=.01)
    B_vectors.append(B)
    print(B*B0,probes[i])

B_vectors = np.array(B_vectors)

# # Setup the PyVista plotter
# plotter = pv.Plotter()

# # Show the probes as spheres
# probe_cloud = pv.PolyData(probes)
# plotter.add_mesh(probe_cloud, color='yellow', point_size=10, render_points_as_spheres=True)

# # Show B field vectors
# plotter.add_arrows(probes, B_vectors, mag=.005, color='blue')

# # Show the wire/coil
# wr.show(forces=None, velocity=False, plotter=plotter)  # Assuming wr.show() adds its parts to a plotter or returns one

# plotter.show()

# Define the bounds of your grid
x_range = (-2.5, 2.5)
z_range = (-3, 8.0)
spacing = 0.2

# Create linearly spaced coordinates
x = np.arange(x_range[0], x_range[1] + spacing, spacing)
z = np.arange(z_range[0], z_range[1] + spacing, spacing)

# Generate 3D meshgrid
X, Z = np.meshgrid(x, z, indexing='xy')
Y = X*0
bx, by, bz = sim.getB(X, Y, Z)


# Compute magnitude (optional, for coloring)
Bmag = B0*np.sqrt(bx**2 + bz**2)

# Plot streamlines
fig, ax = plt.subplots(figsize=(8, 6))

# Use imshow for colormap background (simplifies interactivity)
c = ax.imshow(
    Bmag,         # imshow expects transposed data
    origin='lower',
    extent=(x.min(), x.max(), z.min(), z.max()),
    cmap='plasma',
    aspect='equal'
)
fig.colorbar(c, ax=ax, label='|B|')

# # Colormap background of Bmag
# c = ax.pcolormesh(X, Z, Bmag, shading='auto', cmap='plasma')
# fig.colorbar(c, ax=ax, label='|B|')

strm = plt.streamplot(
    X, Z, bx, bz,
    color='white',  # color by magnitude
    linewidth=1,
    density=1.0
)



plt.xlabel('X')
plt.ylabel('Z')
plt.title('Magnetic Field Streamlines in XZ Plane (Y=0)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

