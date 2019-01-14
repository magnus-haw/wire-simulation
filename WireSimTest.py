import numpy as np
import mayavi.mlab as mlab

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State

import matplotlib.pyplot as plt

### Dimensional scales
r0 = 0.06 #m
L0 = 3.0 #m
I0 = 1000. #Amps
nden0 = 1e18 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s
m0 = rho0*pi*r0*r0*L0
n = 100

print("L0 (m)", L0)
print("B0 (T)", B0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = 0.1
I = 1.
rho = 1.
dm = pi*dr

### Initialize path
phi = np.linspace(pi/6.,5*pi/6.,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]

### Initialize mass
mass = np.ones((n,1))*dm

### Create wire 
wr = Wire(path,path*0,mass,I=I)

### Create intial state
st = State('single_loop_test',load=0)
st.items.append(wr)
#st.save()

### Initialize engine
sim = MultiWireEngine(st,dt)
for i in range(0,300):
    new_st = sim.advance()
new_st.show()

