import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State

import matplotlib.pyplot as plt

### Dimensional scales
L0 = 0.04 #m
r0 = 0.008 #m
I0 = 30000. #Amps
nden0 = 1e22 #m^-3
n = 100

### Derived scales
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*L0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s

### Total mass
loop_len = 2*pi*L0 #m
m0 = rho0*pi*r0*r0*loop_len

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
dt = .001
L = 1.
I = 1.
rho = 1.5
Bp = 1.

r = r0/L0
dm = pi*r*r*(loop_len/L0)*rho/n

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,pi,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
height = L*np.sin(phi)
mass = np.ones((n,1))*np.vstack(np.exp(-height/(0.75*L)))*2*dm

### Create wire 
wr = Wire(path,path*0,mass,I,r=r,Bp=1,is_fixed=False)
##################################################


################ Footpoint coils #################
### Initialize path
phi = np.linspace(0.,2*pi,50)
path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
### Initialize mass
mass = np.ones((len(path0),1))
### Create coils 
coil0 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=r)
coil1 = Wire(path1,path1*0,mass,1,is_fixed=True,r=r)
##################################################


############### Create intial state ##############
st = State('single_loop_test',load=0)
st.items.append(wr)
##st.items.append(coil0)
##st.items.append(coil1)
st.show()
mlab.show()
#st.save()
##################################################


########## Specify Boundary Conditions ###########
def BC(state):
    ### Boundary conditions
    for wire in state.items:
        if not wire.is_fixed:
            # Update current
            wire.I = np.sin(state.time * pi/4.)
            
            # Fix first and final segments
            wire.v[0:2,:]= 0.
            wire.v[-2:,:]= 0.

            # impervious lower boundary
            r0=0.05
            wire.v[2:-2,2][wire.p[2:-2,2] < r0] = 0
            wire.p[2:-2,2][wire.p[2:-2,2] < r0] = r0

            # mass BC
            wire.m[0,0], wire.m[-1,0] = pi*(wire.r)**3
            wire.total_mass = wire.m.sum()
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt,bc=BC)
for i in range(0,1700):
    new_st = sim.advance()

    if i%100 == 0:
##        new_st.show()
##        mlab.show()

        print(new_st.time,new_st.items[0].I)
        myloop = new_st.items[0]
        forces = sim.forceScheme()[0]

        fig = plt.figure(0)
        plt.plot(forces[:,0],forces[:,2])

        fig = plt.figure(1)
        plt.plot(myloop.p[:,0],myloop.p[:,2],'o-')
plt.show()
##################################################


################# Plot Results ###################
plt.figure(0)
plt.title("forces")
forces = sim.forceScheme()[0]
plt.plot(forces[:,0],forces[:,2])

plt.figure(1)
plt.title("position")
wire = sim.state.items[0]
plt.plot(wire.p[:,0],wire.p[:,2],'bo')
plt.show()

##new_st.show()
##mlab.show()
##################################################
