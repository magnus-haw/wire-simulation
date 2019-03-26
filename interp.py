import numpy as np
import mayavi.mlab as mlab
from scipy.interpolate import interp1d,splprep,splev

def parse_path(path):
    x,y,z = path[:,0],path[:,1],path[:,2]
    tck,s = splprep([x,y,z],s=0)
    
    for i in range(0,len(path)):
        path[i] 
