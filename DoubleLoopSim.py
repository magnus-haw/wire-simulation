import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State
from electrodes.CroFT_electrodes import plot_all as plot_electrodes

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
dt = .02
I = 1.
rho = 1.
dm = pi*dr

################### Electrodes ###################
apos,bpos = plot_electrodes(plot_candelabra=False, plot_loops=0)
loop_A_lower_pos,loop_A_upper_pos =apos
loop_B_lower_pos,loop_B_upper_pos =bpos

### Initialize paths
phi = np.linspace(0.,pi,n)
s = np.cos(phi)
one = np.matrix(np.ones((1,len(phi))) ).T
R_0= .5*np.linalg.norm(loop_A_upper_pos - loop_A_lower_pos)/1.3

p0,p1 = loop_A_upper_pos, loop_A_lower_pos
p = np.matrix(s).T*np.matrix((p1-p0)/2.) + one*np.matrix((p1+p0)/2.)
path1 = np.array([p[:,0].A1, p[:,1].A1, R_0*np.sin(phi)]) # Circular

p0,p1 = loop_B_upper_pos, loop_B_lower_pos
p = np.matrix(s).T*np.matrix((p1-p0)/2.) + one*np.matrix((p1+p0)/2.)
path2 = np.array([p[:,0].A1, p[:,1].A1, R_0*np.sin(phi)]) # Circular
##################################################

################ Double loop wires ###############
### Initialize mass
mass = np.ones((n,1))*dm
### Create wire 
wr_a = Wire(path1.T,path1.T*0,mass,I,r=.3,Bp=1)
wr_b = Wire(path2.T,path2.T*0,mass,I,r=.3,Bp=1)
##################################################


################## Footpoint coils #################
##### Initialize path
##phi = np.linspace(0.,2*pi,50)
##path0 = np.array([(L/4)*np.cos(phi)-L,(L/4)*np.sin(phi),0*phi-1]).T
##path1 = np.array([(L/4)*np.cos(phi)+L,(L/4)*np.sin(phi),0*phi-1]).T
##### Initialize mass
##mass = np.ones((len(path0),1))
##### Create coils 
##coil_a1 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
##coil_a2 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
##coil_b1 = Wire(path0,path0*0,mass,-1,is_fixed=True,r=.1)
##coil_b2 = Wire(path1,path1*0,mass,1,is_fixed=True,r=.1)
####################################################



############### Create intial state ##############
st = State('double_loop_test',load=0)
st.items.append(wr_a)
st.items.append(wr_b)
##st.items.append(coil_a1)
##st.items.append(coil_a2)
##st.items.append(coil_b1)
##st.items.append(coil_b2)
st.show()
mlab.show()
#st.save()
##################################################


############## Run simulation engine #############
sim = MultiWireEngine(st,dt)
for i in range(0,500):
    new_st = sim.advance()

    if i%10 == 0:
        new_st.show()
        mlab.show()
        forces = sim.forceScheme()[0]
        plt.plot(forces[:,0],forces[:,2])
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
