import numpy as np
import math
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import sys,os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd + "/classes/")

from Constants import mu0, pi, Kb,amu,mass_elec,elec
from Engine import MultiWireEngine
from Wires import Wire,State
from Utility import biot_savart_SI

from scipy import interpolate

### Dimensional scales
L = .03 #m
I = 1. #A
B0 = mu0*I/(2*pi*L) #Tesla
v0 = 150 #m/s
tau = L/v0 #s
dt = .02*tau
n=1001
mu = pi*4e-7

print("L (m)", L)
print("I (A)", B0)
print("tau (s)", tau)

############################ GENERAL FUNCTIONS #################################

# Construct an array with repeating elements
def array_const(num, len):
    array = []
    for i in range(0, len):
        array.append(num)
    return array

# Returns an array scaled based on the boundaries inputted
def scale(array, range, theta_bool=False):
    return np.interp(array, (array.min(), array.max()), range)

# Returns the arrays of coordinates of n-numbered, equally spaced points on a
# circle with radius R
def unitCircle(num_points, R, theta_bool = False):
    theta_int = 2*np.pi/(num_points)

    theta = 0.
    theta_array = []
    for i in range(num_points):
        theta_array.append(theta)
        theta = theta + theta_int

    if theta_bool == True:
        return theta_array

    x_end = []
    y_end = []
    for i in theta_array:
        x_end.append(R*np.cos(i))
        y_end.append(R*np.sin(i))

    return x_end, y_end

# Produces an array that represents a path that rotates a cone-like or cylindircal shape
def sinusoidalArray(length, radius1, radius2, start_z, end_z, start_angle, end_angle):

    if abs(end_angle - start_angle) > np.pi:
            print('TRUE')
            start_angle = start_angle - 2*np.pi

    z_int = (end_z-start_z)/length
    rad_int = (radius2-radius1)/length
    angle_int = (end_angle-start_angle)/length

    z = []
    angle_array = []
    rad_array = []

    angle = start_angle
    z_inst = start_z
    radius = radius1
    for i in range(length):
        z.append(z_inst)
        angle_array.append(angle)
        rad_array.append(radius)
        z_inst = z_inst + z_int
        radius = radius + rad_int
        angle = angle + angle_int

    x = []
    y = []
    for i in range(len(rad_array)):
        x_inst = rad_array[i]*np.cos(angle_array[i])
        y_inst = rad_array[i]*np.sin(angle_array[i])
        x.append(x_inst)
        y.append(y_inst)

    return np.array(x), np.array(y), np.array(z)

# Produces an array that represents a path with exponential motion
def exponentialArray(length, x1, x2, y1, y2, z1, z2):
    x_int = (x2-x1)/length
    y_int = (y2-y1)/length
    z_int = (z2-z1)/length

    z = []
    y = []
    x = []

    z_inst = z1
    for i in range(length):
        z.append(z_inst)
        z_inst = z_inst + z_int

    if x1 < 0.001 and x1 >= 0. : x1 = 0.001
    if y1 < 0.001 and y1 >= 0. : y1 = 0.001
    if x2 < 0.001 and x2 >= 0. : x2 = 0.001
    if y2 < 0.001 and y2 >= 0. : y2 = 0.001
    cx = np.log(abs(x1))
    cy = np.log(abs(y1))
    kx = (np.log(abs(x2)) - cx)*4.
    ky = (np.log(abs(y2)) - cy)*4.
    for i in range(len(z)):
        sign_x = x2/abs(x2)
        sign_y = y2/abs(y2)
        x_inst = sign_x*np.exp(kx*z[i]+cx)
        y_inst = sign_y*np.exp(ky*z[i]+cy)
        y.append(y_inst)
        x.append(x_inst)

    x, y, z = np.array(x), np.array(y), np.array(z)
    if x2 < x1 and y2 >= y1:
        x, y, z = scale(x, (x2, x1)), scale(y, (y1,
                  y2)), scale(z, (z1, z2))
    elif y2 < y1 and x2 >= x1:
        x, y, z = scale(x, (x1, x2)), scale(y, (y2,
                  y1)), scale(z, (z1, z2))
    elif x2 < x1 and y2 < y1:
        x, y, z = scale(x, (x2, x1)), scale(y, (y2,
                  y1)), scale(z, (z1, z2))
    else:
        x, y, z = scale(x, (x1, x2)), scale(y, (y1,
                  y2)), scale(z, (z1, z2))

    return x, y, z

# Adds a number of last elements of an array to the beginning of an array
def connectArrays(array1, array2):
    last_index5 = len(array1) - 50
    last_elem_array = array1[last_index5:]
    np.insert(array2, 0, last_elem_array)
    return array2

# Function that produces arrays of path1 and path2 with the functions above
def createPaths(num_paths, num_points, x_start, y_start, R_start, R_end, x_end, y_end, z1, z2, z3):
    x1_end, y1_end = unitCircle(num_paths, R_start)

    path1_x = []
    path1_y = []
    path1_z = None
    for i in range(num_paths):
        x, y, z = exponentialArray(num_points, x_start, x1_end[i], y_start, y1_end[i], z1, z2)
        path1_x.append(x)
        path1_y.append(y)
        path1_z = z

    path2_x = []
    path2_y = []
    theta_array = unitCircle(num_paths, R_start, theta_bool = True)
    end_angle = np.arctan(y_end/x_end)
    path2_z = None
    for i in range(num_paths):
        theta_inst = theta_array[i]
        x, y, z = sinusoidalArray(num_points, R_start, R_end, z2, z3, theta_inst, end_angle)
        path2_x.append(x)
        path2_y.append(y)
        path2_z = z

    for i in range(len(path1_x)):
        path2_x[i] = connectArrays(path1_x[i], path2_x[i])
        path2_y[i] = connectArrays(path1_y[i], path2_y[i])

    path2_z = connectArrays(path1_z, path2_z)

    return path1_x, path1_y, path2_x, path2_y, path1_z, path2_z
################################################################################
# Initialize probes
R = 0.07
z = 0.9
probes = np.array([[R*np.cos((5./4.)*pi), R*np.cos((5./4.)*pi), z],
                   [-R*np.cos((5./4.)*pi), R*np.cos((5./4.)*pi), z],
                   [R*np.cos((5./4.)*pi), -R*np.cos((5./4.)*pi), z],
                   [-R*np.cos((5./4.)*pi), -R*np.cos((5./4.)*pi), z]])
wire_rad = 0.0005

#-----------------------------------PATH 1-------------------------------------#

paths1_x, paths1_y, paths2_x, paths2_y, path1_z, path2_z = createPaths(100, 100, 0., 0., 0.035, 0.055, 0.01, 0.055, 0., .8, 1.)

# Use interpolation tool to finalize the paths into wires
wires_1 = []

for i in range(len(paths1_x)):
    x = paths1_x[i]
    y = paths1_y[i]
    z = path1_z
    f_x = interpolate.interp1d(z, x, kind='quadratic')
    f_y = interpolate.interp1d(z, y, kind='quadratic')
    znew = np.linspace(z.min(), z.max(), num=n, endpoint=True)
    xnew = f_x(znew)
    ynew = f_y(znew)
    path = np.array([xnew, ynew, znew]).T
    mass = np.ones((n,1))
    wr = Wire(path, path*0, mass, I, r=wire_rad, Bp=1)
    wires_1.append(wr)

wires_2 = []

for i in range(len(paths2_x)):
    x = paths2_x[i]
    y = paths2_y[i]
    z = path2_z
    f_x = interpolate.interp1d(z, x, kind='quadratic')
    f_y = interpolate.interp1d(z, y, kind='quadratic')
    znew = np.linspace(z.min(), z.max(), num=n, endpoint=True)
    xnew = f_x(znew)
    ynew = f_y(znew)
    path = np.array([xnew, ynew, znew]).T
    mass = np.ones((n,1))
    wr = Wire(path, path*0, mass, I, r=wire_rad, Bp=1)
    wires_2.append(wr)

# Construct power supply path
power_supply_y = np.array([24.8, 27.7, 29.6, 31.3, 32.2, 33.2, 37.5, 41.6, 80.5])
power_supply_z = np.array([19.5, 21.37, 21.4, 20.6, 19.5, 18.1, 11.0, 2.9, -75.8])
power_supply_x = np.array(array_const(0.01, len(power_supply_z)))
power_supply_y = scale(power_supply_y, (paths2_y[0][len(paths2_y[0])-1], 0.8))

# ASSURING THAT PATH 2 AND PATH 3 ARE COINCIDENT
power_supply_z = scale(power_supply_z, (0.2, path2_z[len(path2_z) - 1]))
diff = abs(power_supply_z[0] - path2_z[len(path2_z)-1])
power_supply_z = scale(power_supply_z, (0.2, diff + path2_z[len(path2_z) - 1]))

f_x = interpolate.interp1d(power_supply_y, power_supply_x, kind='quadratic')
f_z = interpolate.interp1d(power_supply_y, power_supply_z, kind='quadratic')
ynew = np.linspace(power_supply_y.min(), power_supply_y.max(), num=n, endpoint=True)
xnew = f_x(ynew)
znew = f_z(ynew)
path = np.array([xnew, ynew, znew]).T
mass = np.ones((n,1))
wr = Wire(path, path*0, mass, I, r=wire_rad, Bp=1)

# Calculate B field at probes[0]
Bvec_x = []
Bvec_y = []
Bvec_z = []
for i in range(len(wires_1)):
    Bvec_x.append(biot_savart_SI(probes[0], I, wires_1[i].p, delta = 0.01)[0])
    Bvec_y.append(biot_savart_SI(probes[0], I, wires_1[i].p, delta = 0.01)[1])
    Bvec_z.append(biot_savart_SI(probes[0], I, wires_1[i].p, delta = 0.01)[2])
    Bvec_x.append(biot_savart_SI(probes[0], I, wires_2[i].p, delta = 0.01)[0])
    Bvec_y.append(biot_savart_SI(probes[0], I, wires_2[i].p, delta = 0.01)[1])
    Bvec_z.append(biot_savart_SI(probes[0], I, wires_2[i].p, delta = 0.01)[2])
    Bvec_x.append(biot_savart_SI(probes[0], I, wr.p, delta = 0.01)[0])
    Bvec_y.append(biot_savart_SI(probes[0], I, wr.p, delta = 0.01)[1])
    Bvec_z.append(biot_savart_SI(probes[0], I, wr.p, delta = 0.01)[2])
    wires_1[i].show()
    wires_2[i].show()

Bvec_x = np.array(Bvec_x)
Bvec_y = np.array(Bvec_y)
Bvec_z = np.array(Bvec_z)

Bvec = [np.mean(Bvec_x), np.mean(Bvec_y), np.mean(Bvec_z)]
print(Bvec)

wr.show() # Needs to be repeated 100 times
mlab.points3d(probes[:,0], probes[:,1], probes[:,2],scale_factor=0.01)
mlab.show()
