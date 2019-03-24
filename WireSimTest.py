import numpy as np
import mayavi.mlab as mlab

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State

import matplotlib.pyplot as plt

### Dimensional scales
r0 = 0.01 #m
L0 = .04 #m
I0 = 10000. #Amps
nden0 = 1e21 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA0 = B0/np.sqrt(mu0*rho0)
tau0 = L0/vA0 #s
m0 = rho0*pi*r0*r0*L0
n = 30

print("L0 (m)", L0)
print("B0 (T)", B0)
print("rho0 (kg/m^3)", rho0)
print("tau (s)", tau0)
print("vA (m/s)", vA0)
print("m (kg)", m0)

### Non-dimensional parameters
L = L0/r0
dr = 1.
dt = .02
I = 1.
rho = 1.
dm = pi*dr

################ Single loop wire ################
### Initialize path
phi = np.linspace(0.,pi,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire 
wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
T,CumLen,dl,N,R = wr.get_3D_curve_params()
wr.L0 = CumLen[-1]
##################################################



################ Footpoint coils #################
### Initialize path
phi = np.linspace(0.,2*pi,20)
path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
### Initialize mass
mass = np.ones((len(path0),1))
### Create coils 
coil0 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
coil1 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
##################################################



############### Create intial state ##############
st = State('single_loop_test',load=0)
st.items.append(wr)
#st.items.append(coil0)
#st.items.append(coil1)
st.show()
mlab.show()
#st.save()
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt)
for i in range(0,1000):
    new_st = sim.advance()

    if i%100 == 0:
        new_st.show()
        mlab.show()
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
