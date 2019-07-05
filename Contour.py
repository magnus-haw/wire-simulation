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
from Utility import biot_savart

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from TimeSeriesSim import timeSeries

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
n = 5000

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
ncl = 21 # number of colors for contours

############ HELPER FUNCTIONS ##############
# Calculate magnitude of a vector
def mag(components):
    x = components[0]
    y = components[1]
    z = components[2]
    magnitude = ((x**2) + (y**2) + (z**2))**(0.5)
    return magnitude

# Calculate the magnetic field's angle phi
def mag_phi(components):
    x = components[0]
    y = components[1]
    z = components[2]
    adj = ((x**2) + (y**2))**(0.5)
    phi = np.arctan(z/adj)
    return phi

# Calculate the magnetic field's angle theta
def mag_theta(components):
    x = components[0]
    y = components[1]
    theta = np.arctan(y/x)
    return theta

# Calculate the angle relative to the column
def angleCol(B, coordinate):
    if coordinate[0] == 0:
        angle = mag_phi(B)
    if coordinate[2] == 0:
        angle = mag_theta(B)
    return angle

#Calculate the angle relative to cross section plane
def angleCross(B, coordinate):
    if coordinate[0] == 0:
        angle = mag_theta(B)
    if coordinate[2] == 0:
        angle = mag_phi(B)
    return angle

#gets the color of the contour
def getColor(cc):
    cl_array = cc.get_facecolor()
    rgba = cl_array[0]
    return rgba

#returns a boolean if the color matches with an element of an array of colors
def compareColors(color, clAr, bool = False):
    if bool:
        if ((clAr[0] == color[0]) and (clAr[1] == color[1]) and (clAr[2] == color[2])):
            return True
    else:
        for i in clAr:
            if ((i[0] == color[0]) and (i[1] == color[1]) and (i[2] == color[2])):
                return True
    return False

# returns the index of which the values falls on relative to the contour's colorbar
def colorNumber(value, boundaries):
    for i in range(len(boundaries)):
        if i == (len(boundaries) - 1):
            return i
        if value < boundaries[i + 1]:
            return i

# returns a Polygon object that defines the shape of a contour
def getShapes(value, contour, colorBar):
    cl_nm = colorNumber(value, colorBar.boundaries)
    cnt_verts = getContourVerts(contour, cl_nm)
    x = arrayConvert(cnt_verts)
    if len(x) > 1:
        polygon_array = []
        for i in x:
            temp = Polygon(i)
            polygon_array.append(temp)
        return polygon_array
    return [Polygon(x[0])]

# converts a nested array to a 'cleaner' array
def arrayConvert(shapeArray):
    array = shapeArray[0]
    if len(array) == 1:
        poly_array = []
        for i in array[0]:
            x = i[0]
            y = i[1]
            poly_array.append((x,y))
        return [poly_array]
    else:
        poly_array = []
        for i in array:
            poly_i = []
            for j in i:
                x = j[0]
                y = j[1]
                poly_i.append((x,y))
            poly_array.append(poly_i)
        return poly_array

# returns an array of points that defines a contour's shape
def getContourVerts(contour, cl_nm):
    contour_v = []
    colorArray = []
    shapesArray = []
    for cc in contour.collections:
        paths = []
        cl = getColor(cc)
        if not (compareColors(cl, colorArray)):
            colorArray.append(cl)
        for pp in cc.get_paths():
            xy = []
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contour_v.append(paths)

    for cc in contour.collections:
        paths = []
        if cl_nm > (len(colorArray) - 1):
            cl_nm = (len(colorArray) - 1)
        if compareColors(colorArray[cl_nm], getColor(cc), True):
            for pp in cc.get_paths():
                xy = []
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                paths.append(np.vstack(xy))
            shapesArray.append(paths)
    return shapesArray

# Finds any intersection between three shapes
# REMINDER: Rather than finding the largest area of col independent of cross,
# you should find any common intersections that have shared area
def intersection(shape_col, shape_mag, shape_cross):
    # At the moment, the 'mag' contour has a list of only unique shapes
    poly_mag = shape_mag[0]
    poly_col = None
    poly_cross = None

    if len(shape_col) > 1:
        viable_poly_col = []
        # Find all viable polygons that intersect with poly_mag
        for i in shape_col:
            if i.intersects(poly_mag):
                viable_poly_col.append(i)

        if len(viable_poly_col) == 1:
            poly_col = viable_poly_col[0]

        if len(viable_poly_col) > 1:
            # Find polygon with largest intersection area
            poly_col = probPoly(poly_mag, viable_poly_col)

    if len(shape_col) == 1:
        poly_col = shape_col[0]

    if len(shape_cross) > 1:
        viable_poly_cross = []
        # Find all viable polygons that intersect with poly_mag
        for i in shape_cross:
            if i.intersects(poly_mag):
                viable_poly_cross.append(i)

        if len(viable_poly_cross) == 1:
            poly_cross = viable_poly_cross[0]

        if len(viable_poly_cross) > 1:
            # Find polygon with largest intersection area
            poly_cross = probPoly(poly_mag, viable_poly_cross)

    if len(shape_cross) == 1:
        poly_cross = shape_cross[0]

    return [poly_col, poly_cross, poly_mag]

def getLambdaAndRadius(shape_col, shape_mag, shape_cross):
    poly_array = intersection(shape_col, shape_mag, shape_cross)

    center_col = poly_array[0].centroid.coords[0]
    center_mag = poly_array[1].centroid.coords[0]
    center_cross = poly_array[2].centroid.coords[0]
    print('Centroid of col = ' + str(center_col))
    print('Centroid of mag = ' + str(center_mag))
    print('Centroid of cross = ' + str(center_cross))

    mean_lambda = (center_col[0] + center_mag[0] + center_cross[0])/3.
    mean_radius = (center_col[1] + center_mag[1] + center_cross[1])/3.

    return [mean_lambda, mean_radius]

# Returns the most probable polygon based on its intersection's area
def probPoly(poly_mag, viable_poly_array):
    prob_poly = viable_poly_array[0]
    max_area = viable_poly_array[0].intersection(poly_mag)

    for i in viable_poly_array:
        area = i.intersection(poly_mag)
        if area > max_area:
            max_area = area
            prob_poly = i

    return prob_poly
############################################
#Calculate spherical coordinates of time-series of magnetic field

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharex = True)
B_data = timeSeries(45., 3., 100, 0., 0., 0.)
ax1.plot(B_data[0])
ax1.set_ylabel('Magnitude of Magnetic Field Vector')
ax1.set_xlabel('Time (Samples)')
ax2.plot(B_data[1])
ax2.set_ylabel('Angle Relative to Column')
ax2.set_xlabel('Time (Samples)')
ax3.plot(B_data[2])
ax3.set_ylabel('Angle Relative to Cross Section')
ax3.set_xlabel('Time (Samples)')

################################################################################
#Construct the contour arrays and colorbars

#Initialize path
phi = np.linspace(-48*pi, 48*pi, n)
path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
### path[:,1] -= path[0,1] ### ASK MAGNUS FOR PURPOSE
#Initialize mass
mass = np.ones((n, 1))**dm
#Create wire
wr = Wire(path, path*0., mass, I, r=0.3, Bp=1)

#Initialize
l = 15*2*pi
rad = 4.67
res = 26
l_rng = range(1, res)
r_rng = range(1, res)

#Create a series of points
#NOTE that the distances may not be to scale as in AHF
probes = np.array([[rad*L, 0 , 0], [0, 0 , rad*L], [-rad*L, 0, 0],
                   [0, 0 , -L*rad], [rad*L, l*13 , 0], [0, l*13, rad*L],
                   [-rad*L, l*13 , 0], [0, l*13, -rad*L]])

# Plot probes and 'wire' of current
mlab.points3d(probes[:,0], probes[:,1], probes[:,2],scale_factor=4)
mlab.points3d(0, 0, 0, scale_factor=4, color=(1., 0., 0.))
wr.show()
mlab.show()

# Initialize grid arrays for the contour
l_array = []
r_array = []

#Initialize contour-value arrays
bmag = []
bangleCol = []
bangleCross = []

#Iterate and produce gird array for helix's radius
for i in r_rng:
    #The constant allows the radius to vary between twice its current size to near 0
    constant = 0.04*i
    #append to the grid array the actual values of the varied radius
    r_array.append(2.*L*constant)

for i in l_rng:
    bmag_i = []
    bangleCol_i = []
    bangleCross_i = []
    for j in r_rng:
        const1 = 0.04*i
        const2 = 0.04*j
        phi = np.linspace(-48*pi,48*pi,n)
        path = np.array([2.*L*np.cos(phi)*const2,1.5*15*phi*const1,2.*L*np.sin(phi)*const2]).T
        #path[:,1] -= path[0,1]
        wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

        # print('Radius of wire: ' + str(2.*L*const2))
        # print('Lambda of wire: ' + str(1.5*l*const1))

        #Calculate mangetic field at probe location
        B = biot_savart(probes[0], I, wr.p, delta = 0.1)
        #Append values to one dimensional array
        bmag_i.append(mag(B))
        bangleCol_i.append(angleCol(B, probes[0]))
        bangleCross_i.append(angleCross(B, probes[0]))

    #append to the grid array the actual values of the varied lambdas
    l_array.append(1.5*l*const1)
    #Append 1D arrays to make two dimensional arrays
    bmag.append(bmag_i)
    bangleCol.append(bangleCol_i)
    bangleCross.append(bangleCross_i)

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharex = True)

# Plot contour map of angle relative to Column
CS0 = ax1.contourf(l_array, r_array, bangleCol, ncl)
CB0 = fig.colorbar(CS0, shrink=1., extend='both', ax=ax1)
ax1.set_ylabel('Radius of Helix Current [cm]')
ax1.set_xlabel('Lambda [cm]')
ax1.title.set_text('Angle Relative to Column vs. L & R')

# Plot contour map of magnitude of magnetic field vector
CS1 = ax2.contourf(l_array, r_array, bmag, ncl)
CB1 = fig.colorbar(CS1, shrink=1., extend='both', ax=ax2)
ax2.set_ylabel('Radius of Helix Current [cm]')
ax2.set_xlabel('Lambda [cm]')
ax2.title.set_text('Magnitude of Magnetic Field Vector vs. L & R')

# Plot contour map of angle relative to Column
CS2 = ax3.contourf(l_array, r_array, bangleCross, ncl)
CB2 = fig.colorbar(CS2, shrink=1., extend='both')
ax3.set_ylabel('Radius of Helix Current [cm]')
ax3.set_xlabel('Lambda [cm]')
ax3.title.set_text('Angle Relative to Cross Section vs. L & R')

################################################################################
# Identify contour levels depending on the Magnetic Field vector

# Coordinate arrays
coord_x = []
coord_y = []

for i in range(len(bmag)):
    mag = B_data[0][i]
    col = B_data[1][i]
    cross = B_data[2][i]

    print('col Value = ' + str(col))
    print('mag Value = ' + str(mag))
    print('cross Value = ' + str(cross))

    shape_col = getShapes(col, CS0, CB0)
    shape_mag = getShapes(mag, CS1, CB1)
    shape_cross = getShapes(cross, CS2, CB2)
    print(getLambdaAndRadius(shape_col, shape_mag, shape_cross))
    print('\n')

plt.show()
