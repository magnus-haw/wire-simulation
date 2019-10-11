import time
import numpy as np

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State
import matplotlib.pyplot as plt

### Non-dimensional parameters
n=50
L = 10 #units == minor radii
dt = .01
I = 1.
rho = 1.

dm = pi*L*rho/n

############### Single half circle ###############
### Initialize path
phi = np.linspace(0.,pi,n)
path = np.array([L*np.cos(phi),0*phi,L*np.sin(phi)]).T
path[:,1] -= path[0,1]
### Initialize mass
mass = np.ones((n,1))

### Create wire 
wr = Wire(path,path*0,mass,I,r=1,Bp=1,is_fixed=False)
##################################################

############### Create intial state ##############
st = State('benchmark_single_wire',load=0)
st.items.append(wr)
#st.save()
##################################################

############## Run simulation engine #############
sim = MultiWireEngine(st,dt)
start_time = time.time()
for i in range(0,int(10/dt)):
    new_st = sim.advance()
    if i%50 == 0:
##        new_st.show()
##        mlab.show()

        myloop = new_st.items[0]
        forces = sim.forceScheme()[0]

        fig = plt.figure(0)
        plt.plot(forces[:,0],forces[:,2])

        fig = plt.figure(1)
        plt.plot(myloop.p[:,0],myloop.p[:,2],'o-')
end_time = time.time()
wire = new_st.items[0]
print("npoints= %i, nwires= %i, nsteps = %i, Runtime= %.2f"%(len(wire.p),
                                                             len(new_st.items),
                                                             int(10/dt),
                                                             end_time - start_time) )
##plt.show()
##################################################









################## Footpoint coils #################
##### Initialize path
##phi = np.linspace(0.,2*pi,50)
##path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
##path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
##### Initialize mass
##mass = np.ones((len(path0),1))
##### Create coils 
##coil0 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
##coil1 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
####################################################
