#Make spheromak movie
import numpy as np
import mayavi.mlab as mlab
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")
sys.path.append(cwd + "/Dropbox/mylib/")

from makeMovie import Frame,FrameList
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

Load_from_file =1
start = 0.00000
end   = 1.33000
dt = .01
times = np.arange(start,end,dt)

make_frames=1
annotate_frames=0
make_movie =0

if make_frames:
    for i in range(0,len(times)):
        Time_to_load = times[i]
        frame_name = "frame{:0>4}.png".format(i)

        st = State('spheromak_test',time=Time_to_load, load=Load_from_file)
        st.show(velocity=True)

        ### Grab figure object handles
        figure = mlab.gcf()
        engine = mlab.get_engine()
        scene = engine.scenes[0]

        ### Set camera angle
        scene.scene.camera.position = [-9.098904693636317, -16.545415296822437, 11.108752373259385]
        scene.scene.camera.focal_point = [-0.04371798999832638, -0.261976800211929, 1.8559575817610865]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [0.2059041025246023, 0.39439365641753693, 0.8955764313229462]
        scene.scene.camera.clipping_range = [5.415953452359608, 39.86293122609016]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        ### Save frame
        mlab.savefig(frame_name,size=(500,500),figure=figure)
        mlab.clf()

fontdir = '/usr/share/fonts/truetype/msttcorefonts/'

if annotate_frames:
    prefix = 'frame'
    path = '/home/magnus/Desktop/wire-simulation/jet_case_0/'
    fl = FrameList(path,prefix,'.png',(2,133),output="annotated/")
    fl.annotate('Initialize wires',(45,45), (2,3))
    fl.annotate('Calculate magnetic forces',(45,45), (3,4))
    
if make_movie:
    prefix = 'frame'
    path = '/home/magnus/Desktop/wire-simulation/jet_case_0/'
    fl = FrameList(path,prefix,'.png',(2,133),output="movie/")
    fl.resize((492,492))
    fl.make_movie()
