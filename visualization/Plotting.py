### Visualization functions

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
from electrodes.ARCHeS_electrodes import torus

def plot_force(path,f):
    plt.plot(path[:,0],path[:,1],'bo')
    plt.quiver(path[2:-2,0],path[2:-2,1],f[:,0],f[:,1])
    plt.show()
    
def plot_path(path):
    mlab.plot3d(path[:,0], path[:,1], path[:,2], tube_radius=r0/2., color=(1,0,0))
