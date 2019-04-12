import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State
from electrodes.jet_electrodes import get_jet_nozzles,annulus_electrode,center_electrode

import matplotlib.pyplot as plt

### Dimensional scales
r0 = 0.01 #m
L0 = .2 #m
I0 = 10000. #Amps
nden0 = 1e21 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s
m0 = rho0*pi*r0*r0*L0
n = 11

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = .01
I = 1.
B = 1.
rho = 1.
dm = pi*dr

################### Electrodes ###################
annulus_electrode()
center_electrode()
inner,outer = get_jet_nozzles()
phi = np.linspace(0.,pi,n)

### Initialize mass
mass = np.ones((n,1))*dm

### Initialize paths
mywires=[]
for i in range(0,8):
    start,end = np.array(inner[i]),np.array(outer[i])
    pvec = end-start
    st = np.zeros(3)
    st[0:2] = end
    R0 = np.linalg.norm(pvec)/2.
    y,z = R0*np.cos(phi),R0*np.sin(phi)
    yaxis,zaxis = np.zeros((3,1)),np.zeros((3,1))
    yaxis[1,0],zaxis[2,0] = pvec[1]/(R0*2), 1.
    yaxis[0,0] = pvec[0]/(R0*2)

    path = ((y-y[0])*yaxis + z*zaxis).T + st
    mlab.plot3d(path[:,0], path[:,1], path[:,2], tube_radius=r0, color=(1-.001*i,.001*i,.001*i))

    mywires.append(Wire(path/L0,path*0,mass,I,r=.1,Bp=B))
mlab.show()

############## Background solenoid ###############
### Initialize path
phi = np.linspace(0.,2*pi,50)
mass = np.ones((len(phi),1))
path = np.array([0.2*np.cos(phi)/L0,0.2*np.sin(phi)/L0,0*phi-.01]).T
coil = Wire(path,path*0,mass,-1,is_fixed=True,r=.1)
##################################################



############### Create intial state ##############
st = State('spheromak_test',load=0)
for w in mywires:
    st.items.append(w)
st.items.append(coil)
st.show()
mlab.show()
#st.save()
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt)
for i in range(0,500):
    new_st = sim.advance()

    if i%50 == 0:
        new_st.show()
        mlab.show()
        forces = sim.forceScheme()[0]
        plt.plot(forces[:,1],forces[:,2])
        plt.show()
##################################################


################# Plot Results ###################
plt.figure(0)
plt.title("forces")
forces = sim.forceScheme()[0]
plt.plot(forces[:,1],forces[:,2])

plt.figure(1)
plt.title("position")
wire = sim.state.items[0]
plt.plot(wire.p[:,1],wire.p[:,2],'bo')
plt.show()

##new_st.show()
##mlab.show()
##################################################
