import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire
from State import State
from electrodes.CroFT_electrodes import plot_all as plot_electrodes

import matplotlib.pyplot as plt

### Dimensional scales
L0 = 0.075 #m
r0 = 0.020 #m
I0 = 20000. #Amps
nden0 = 0.75e21 #m^-3
n = 100

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

################### Electrodes ###################
apos,bpos = plot_electrodes(plot_candelabra=False, plot_loops=0)
loop_A_lower_pos,loop_A_upper_pos =apos[0]/L0,apos[1]/L0
loop_B_lower_pos,loop_B_upper_pos =bpos[0]/L0,bpos[1]/L0

### Initialize normalized path positions
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
height = R_0*np.sin(phi)
mass = np.ones((n,1))*np.vstack(np.exp(-height/(0.75*L)))*2*dm
### Create wire 
wr_a = Wire(path1.T,path1.T*0,mass,I,r=r,Bp=Bp)
wr_b = Wire(path2.T,path2.T*0,mass,I,r=r,Bp=Bp)
##################################################


################ Footpoint coils #################
### Initialize path
phi = np.linspace(0.,2*pi,50)
mass = np.ones((len(phi),1))
solenoids =[]
for footpoint in [loop_A_lower_pos,loop_B_lower_pos]:
    xpos, ypos = footpoint
    path = np.array([(L/4)*np.cos(phi)+xpos,(L/4)*np.sin(phi)+ypos,0*phi-.1]).T
    coil = Wire(path,path*0,mass,-.1,is_fixed=True,r=.01)
    solenoids.append(coil)
for footpoint in [loop_A_upper_pos,loop_B_upper_pos]:
    xpos, ypos = footpoint
    path = np.array([(L/4)*np.cos(phi)+xpos,(L/4)*np.sin(phi)+ypos,0*phi-.1]).T
    coil = Wire(path,path*0,mass,.1,is_fixed=True,r=.01)
    solenoids.append(coil)
##################################################



############### Create intial state ##############
st = State('double_loop_test',load=0)
st.items.append(wr_a)
st.items.append(wr_b)
##for sol in solenoids:
##    st.items.append(sol)
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
            
            # Fix first and final segments
            wire.v[0:2,:]= 0.
            wire.v[-2:,:]= 0.

            # impervious lower boundary
            r0=0.05
            wire.v[2:-2,2][wire.p[2:-2,2] < r0] = 0
            wire.p[2:-2,2][wire.p[2:-2,2] < r0] = r0
##################################################
            

############## Run simulation engine #############
sim = MultiWireEngine(st,dt,bc=BC)
for i in range(0,1000):
    new_st = sim.advance()

    if i%100 == 0:
        print(new_st.time,new_st.items[0].I)
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
