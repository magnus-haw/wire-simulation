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
n = 1000

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
ncl = 251 # number of colors for contours

###### General functions ######

# Calculate the magnitude of magnetic field w/ its components
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
def angleCrossSection(B, coordinate):
    if coordinate[0] == 0:
        angle = mag_theta(B)
    if coordinate[2] == 0:
        angle = mag_phi(B)
    return angle

# returns a list of all the angles of the magnetic field based on the position
def angleList(B_array, coordinate):
    ang_list = []
    if coordinate[0] == 0:
        for i in range(len(B_array)):
            angle = mag_phi(B_array[i])
            ang_list.append(angle)
    if coordinate[2] == 0:
        for i in range(len(B_array)):
            angle = mag_theta(B_array[i])
            #print(angle)
            #print(B_array[i][0])
            ang_list.append(angle)
    #print(ang_dict)
    return ang_list

# returns the max angle of the magnetic field using angleList()
def maxAngle(B_array, coordinate):
    ang_dict = angleDict(B_array, coordinate)
    max = list(ang_dict.values())[0]
    for i in ang_dict.values():
        if i > max:
            max = i
    B = list(ang_dict.keys())[list(ang_dict.values()).index(max)]
    #print('The max angle is ' + str(max) + ' at the magnetic vector ' + str(B))
    return [B, max]

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

# returns the min and max of an array
def getMinMax(nested_array):
    array = nested_array[0][0]
    min_x = array [0][0]
    min_y = array[0][1]
    max_x = array[0][0]
    max_y = array[0][1]
    for i in range(len(array)):
        if array[i][0] < min_x:
            min_x = array[i][0]
        if array[i][0] > max_x:
            max_x = array[i][0]

        if array[i][1] < min_y:
            min_y = array[i][1]
        if array[i][1] > max_y:
            max_y = array[i][1]
    return [min_x, max_x, min_y, max_y]

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
    x = arrayConvert(getContourVerts(contour, cl_nm))
    return Polygon(x)

# converts a nested array to a 'cleaner' array
def arrayConvert(shapeArray):
    array = shapeArray[0][0]
    poly_array = []
    for i in array:
        x = i[0]
        y = i[1]
        poly_array.append((x,y))
    return poly_array

# returns centroid depending whether there are one or multiple areas of intersection
def center(shape1, shape2):
    poly = shape1.intersection(shape2)
    if not isinstance(poly, MultiPolygon): #if not a MultiPolygon
        return poly.centroid.coords[0]
    else:
        center1 = shape1.centroid.coords[0]
        center2 = shape2.centroid.coords[0]
        mean_x = (center1[0] + center2[0])/2.
        mean_y = (center1[1] + center2[1])/2.
        return [mean_x, mean_y]

# returns midpoint of two points
def avg(center1, center2):
    mean_x = (center1[0] + center2[0])/2.
    mean_y = (center1[1] + center2[1])/2.
    return [mean_x, mean_y]

# returns the centroid of three shapes depedning on several scenarios (check exampel below)
def intersection(shape1, shape2, shape3):
    # does shape1 intersect with shape2
    if shape1.intersects(shape2):
        c = center(shape1, shape2)
        poly = shape1.intersection(shape2)
        #does shape3 intersect with area of intersection of shape1 and shape2
        if poly.intersects(shape3):
            # return center of the three way intersection
            return center(poly, shape3)
        if shape1.intersects(shape3):
            c1 = center(shape1, shape3)
            return avg(c, c1)
        if shape2.intersects(shape3):
            c2 = center(shape2, shape3)
            return avg(c ,c2)
        else:
            c3 = shape3.centroid.coords[0]
            return avg(c, c3)
    if shape2.intersects(shape3):
        c = center(shape2, shape3)
        poly1 = shape2.intersection(shape3)
        if shape1.intersects(shape3):
            c1 = center(shape1, shape3)
            return avg(c, c1)
        else:
            c2 = shape1.centroid.coords[0]
            return avg(c, c2)
    if shape1.intersects(shape3):
        c = center(shape1, shape3)
        poly2 = shape1.intersects(shape3)
        c1 = shape3.centroid.coords[0]
        return avg(c, c1)
    # no intersections whatsoever
    else:
        center1 = shape1.centroid.coords[0]
        center2 = shape2.centroid.coords[0]
        center3 = shape3.centroid.coords[0]
        mean_x = (center1[0] + center2[0] + center3[0])/2.
        mean_y = (center1[1] + center2[1] + center3[1])/2.
        return [mean_x, mean_y]

# returns percent error of the estimated value of the wavelength and radius based on the given 'data'
def modelLambdaRadius(B_array, AngleCol_array, AngleCross_array, B_array1, AngleCol_array1, AngleCross_array1, lmbda, radius):
    # select lambda and radius resolution (note that they have to be same numer)
    cntr_array = twoDimContour(31, 31)

    #Col
    CS = cntr_array[0]
    CS1 = cntr_array[1]
    CB = cntr_array[2]
    #Cross
    ES = cntr_array[3]
    ES1 = cntr_array[4]
    EB = cntr_array[5]
    #Mag
    DS = cntr_array[6]
    DS1 = cntr_array[7]
    DB = cntr_array[8]

    #coordinates arrays
    coord_x = []
    coord_y = []
    coord_x1 = []
    coord_y1 = []
    coord_xtot = []
    coord_ytot = []

    for i in range(len(B_array)):
        col = AngleCol_array[i]
        cross = AngleCross_array[i]
        bmag = B_array[i]
        col1 = AngleCol_array1[i]
        cross1 = AngleCross_array1[i]
        bmag1 = B_array1[i]
        coord = intersection(getShapes(col, CS, CB), getShapes(cross, ES, EB), getShapes(bmag, DS, DB))
        coord1 = intersection(getShapes(col1, CS1, CB), getShapes(cross1, ES1, EB), getShapes(bmag1, DS1, DB))
        coord_x.append(coord[0])
        coord_y.append(coord[1])
        coord_x1.append(coord1[0])
        coord_y1.append(coord1[1])
        c_xtot = (coord[0] + coord1[0])/2.
        c_ytot = (coord[1] + coord1[1])/2.
        coord_xtot.append(c_xtot)
        coord_ytot.append(c_ytot)

    p = percentError(coord_x, coord_y, lmbda, radius)
    print('PROBES[0], for lambda = ' + str(lmbda) + ' and radius = ' + str(radius))
    print('Percent Error for lambda: ' + str(p[0]) + '%')
    print('Percent Error for radius: ' + str(p[1]) + '%')
    print('\n')

    p1 = percentError(coord_x1, coord_y1, lmbda, radius)
    print('PROBES[1], for lambda = ' + str(lmbda) + ' and radius = ' + str(radius))
    print('Percent Error for lambda: ' + str(p1[0]) + '%')
    print('Percent Error for radius: ' + str(p1[1]) + '%')
    print('\n')

    p_tot = percentError(coord_xtot, coord_ytot, lmbda, radius)
    print('For lambda = ' + str(lmbda) + ' and radius = ' + str(radius))
    print('Percent Error (average) for lambda: ' + str(p_tot[0]) + '%')
    print('Percent Error (average) for radius: ' + str(p_tot[1]) + '%')

    return [p[0], p[1], p1[0], p1[1], p_tot[0], p_tot[1]]

# returns the contour arrays and colorbars
def twoDimContour(lmda_res = 101, L_res = 101):

    ################ Single loop wire ################
    ### Initialize path
    phi = np.linspace(0.,36*pi,n) + 0.5
    path = np.array([L*np.cos(phi),15*phi,L*np.sin(phi)]).T
    path[:,1] -= path[0,1]
    ### Initialize mass
    mass = np.ones((n,1))*dm
    ### Create wire
    wr = Wire(path,path*0,mass,I,r=.3,Bp=1)
    ##################################################
    # Initialize length of grid arrays
    lmda = 15*2*pi
    rad = 4.67
    lmda_scl = range(1, lmda_res) #NOTE: both ranges must be the same size
    L_scl = range(1, L_res)

    # Create a series of points NOTE: probes[0] and probes[1] were moved to exactly the middle
    probes = np.array([[4.*L, lmda*9 , 0], [0, lmda*9 , 4.], [0, lmda*9, -4.]])

    # Initialize the grid arrays for the contour
    lambda_array = []
    radius_array = []
    B_array = []
    angleCol_array = []
    angleCross_array = []
    B_array1 = []
    angleCol_array1 = []
    angleCross_array1 = []

    # Iterate and produce grid array for helix's radius
    for k in L_scl:
        phi = np.linspace(0.,36*pi,n) + np.pi
        path = np.array([L*np.cos(phi)/k,15*phi,L*np.sin(phi)/k]).T
        path[:,1] -= path[0,1]
        wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

        radius_array.append(L/k)

    # Iterate through the grid arrays and calculate magnetic field and phi and theta
    for j in lmda_scl:
        B_L = []
        angleCol_L = []
        angleCross_L = []
        B_L1 = []
        angleCol_L1 = []
        angleCross_L1 = []
        for k in L_scl:
            phi = np.linspace(0.,36*pi,n) + np.pi
            path = np.array([L*np.cos(phi)/k,15*phi/j,L*np.sin(phi)/k]).T
            path[:,1] -= path[0,1]
            wr = Wire(path,path*0,mass,I,r=.3,Bp=1)

            B = biot_savart(probes[1], I, wr.p, delta = 0.1)
            B1 = biot_savart(probes[2], I, wr.p, delta = 0.1)
            B_L.append(mag(B))
            B_L1.append(mag(B1))
            angleCol_L.append(angleCol(B, probes[1]))
            angleCol_L1.append(angleCol(B1, probes[2]))
            angleCross_L.append(angleCrossSection(B, probes[1]))
            angleCross_L1.append(angleCrossSection(B1, probes[2]))

        lambda_array.append(lmda/j) # Produce gird array for the helix's lambda
        B_array.append(B_L)
        angleCol_array.append(angleCol_L)
        angleCross_array.append(angleCross_L)
        B_array1.append(B_L1)
        angleCol_array1.append(angleCol_L1)
        angleCross_array1.append(angleCross_L1)

    # Plot contour map of angle relative to column
    fig = plt.figure(3)
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    CS = ax.contourf(lambda_array, radius_array, angleCol_array, ncl)
    CS1 = ax1.contourf(lambda_array, radius_array, angleCol_array1, ncl)
    CB = fig.colorbar(CS, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Angle of Magnetic Field Vector Relative to the Column')

    # Plot contour map of angle relative to cross section
    fig = plt.figure(4)
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ES = ax.contourf(lambda_array, radius_array, angleCross_array, ncl)
    ES1 = ax1.contourf(lambda_array, radius_array, angleCross_array1, ncl)
    EB = fig.colorbar(ES, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Angle of Magnetic Field Vector Relative to the Cross Section Plane')

    # Plot contour map of magnitude of magnetic field
    fig = plt.figure(5)
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    DS = ax.contourf(lambda_array, radius_array, B_array, ncl)
    DS1 = ax1.contourf(lambda_array, radius_array, B_array1, ncl)
    DB = fig.colorbar(DS, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Magnitude of Magnetic Field')

    plt.show()

    return [CS, CS1, CB, ES, ES1, EB, DS, DS1, DB]

# calculates percent error from predicted values and actual value
def percentError(x_array, y_array, x_actual, y_actual):
    x_avg = np.mean(x_array)
    y_avg = np.mean(y_array)
    percent_x = (x_actual - x_avg)*100/x_actual
    percent_y = (y_actual - y_avg)*100/y_actual
    return [percent_x, percent_y]

"""HEY MAGNUS, read the comments below:"""
######## Simulate data similar to AHF runs ###########

# default values of lambda and the radius when looping through one and not the other
# example, when varying lambda the radius will be 2.0 cm
l = 45.
r = 2.

#Pick how much noise you want for each axis
x_scl = 0.1
y_scl = 0.1
z_scl = 0.1

#Pick number for range of time
t = 100

# pick a range of values for lambda and the radius in which it will loop through
l_range = list(range(4, 24))
r_range = list(range(1, 21))

######## Estimate the lamda and radius of helix ######

#for probes[0] (along x-axis)
el_array = []
er_array = []
#for probes[1] (along z-axis)
el_array1 = []
er_array1 = []
#for average of first two probes
el_arraytot = []
er_arraytot = []

# looping through values of different lambda
for i in l_range:
    B_data = timeSeries(i*5., r, t, x_scl, y_scl, z_scl)
    e_array = modelLambdaRadius(B_data[0], B_data[1], B_data[2], B_data[3], B_data[4], B_data[5], i*5., r)
    el_array.append(e_array[0])
    el_array1.append(e_array[2])
    el_arraytot.append(e_array[4])

# looping through values of different radius
for j in r_range:
    B_data = timeSeries(l, j/5., t, x_scl, y_scl, z_scl)
    e_array = modelLambdaRadius(B_data[0], B_data[1], B_data[2], B_data[3], B_data[4], B_data[5], l, j/5.)
    er_array.append(e_array[1])
    er_array1.append(e_array[3])
    er_arraytot.append(e_array[5])

# rescaling the x-axis arrays
for i in range(len(l_range)):
    l_range[i] = l_range[i]*5.

for i in range(len(r_range)):
    r_range[i] = r_range[i]/5.

plt.figure(1)
plt.plot(l_range, el_array, 'ro', l_range, el_array1, 'bo', l_range, el_arraytot, 'go')
plt.ylabel('Percent Error')
plt.xlabel('Lambda [cm]')

plt.figure(2)
plt.plot(r_range, er_array, 'ro', r_range, er_array1, 'bo', r_range, er_arraytot, 'go')
plt.ylabel('Percent Error')
plt.xlabel('Radius [cm]')

plt.show()
