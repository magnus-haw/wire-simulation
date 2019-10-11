import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State
from electrodes.jet_electrodes import get_jet_nozzles,annulus_electrode,center_electrode

import matplotlib.pyplot as plt

### Dimensional scales
L0 = 0.075 #m
r0 = 0.015 #m
I0 = 50000. #Amps
nden0 = 5e20 #m^-3
n = 11

### Derived scales
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*L0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s

### Total mass
loop_len = 0.386 #m
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
rho = 1.
Bp = 1.

r = r0/L0
dm = pi*r*r*(loop_len/L0)*rho/n



Load_from_file =1
Time_to_load = 1.00100
################ Initial Conditions ################
if not Load_from_file:
    annulus_electrode()
    center_electrode()
    inner,outer = get_jet_nozzles()
    phi = np.linspace(0.,pi,n)

    ### Initialize paths
    mywires=[]
    for i in range(0,8):
        start,end = np.array(inner[i]),np.array(outer[i])
        pvec = end-start
        st = np.zeros(3)
        st[0:2] = end
        R0 = np.linalg.norm(pvec)/2.
        y,z = R0*np.cos(phi),R0*np.sin(phi)
        mass = np.ones((n,1))*np.vstack(np.exp(-z/(0.75*L)))*2*dm
        yaxis,zaxis = np.zeros((3,1)),np.zeros((3,1))
        yaxis[1,0],zaxis[2,0] = pvec[1]/(R0*2), 1.
        yaxis[0,0] = pvec[0]/(R0*2)

        path = ((y-y[0])*yaxis + z*zaxis).T + st
        mlab.plot3d(path[:,0], path[:,1], path[:,2], tube_radius=r0, color=(1-.001*i,.001*i,.001*i))

        mywires.append(Wire(path/L0,path*0,mass,I,r=r,Bp=Bp))
    mlab.show()

    ################ Background solenoid ###############
    ##### Initialize path
    ##phi = np.linspace(0.,2*pi,50)
    ##mass = np.ones((len(phi),1))
    ##path = np.array([0.2*np.cos(phi)/L0,0.2*np.sin(phi)/L0,0*phi-.01]).T
    ##coil = Wire(path,path*0,mass,-1,is_fixed=True,r=.1)
    ####################################################



################ Load intial state ###############
st = State('spheromak_test',time=Time_to_load, load=Load_from_file)

if not Load_from_file:
    for w in mywires:
        st.items.append(w)
    #st.items.append(coil)
    st.save()
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
            wire.I = np.sin(state.time * pi/2.)
            wire.r = .15

##            # Smoothing
##            for i in range(0,6):
##                wire.smooth()
            
            # Fix first and final segments
            wire.v[0:2,:]= 0.
            wire.v[-2:,:]= 0.

            # impervious lower boundary
            r0=0.05
            wire.v[2:-2,0][wire.p[2:-2,2] <= r0] = 0
            wire.v[2:-2,1][wire.p[2:-2,2] <= r0] = 0
            wire.v[2:-2,2][wire.p[2:-2,2] <= r0] = 0
            wire.p[2:-2,2][wire.p[2:-2,2] <= r0] = r0-.0001

            # mass BC
            wire.m[0,0], wire.m[-1,0] = pi*(wire.r)**3, pi*(wire.r)**3
            wire.total_mass = wire.m.sum()
##################################################

############## Run simulation engine #############
sim = MultiWireEngine(st,dt,bc=BC)
for i in range(1,100):
    new_st = sim.advance()
    if i%1 == 0:
        print(i,new_st.time,new_st.items[0].I)
        print(new_st.items[0].v[:,2].max())
        plt.plot(new_st.items[0].v[:,2])
        plt.show()
        new_st.show()
        mlab.show()

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
