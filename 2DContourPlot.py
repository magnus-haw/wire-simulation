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
from shapely.geometry import Polygon
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
    #print('Length: ' + str(len(colorArray)) + ' and array: ' + str(colorArray))

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

def getColor(cc):
    cl_array = cc.get_facecolor()
    rgba = cl_array[0]
    return rgba

def compareColors(color, clAr, bool = False):
    if bool:
        if ((clAr[0] == color[0]) and (clAr[1] == color[1]) and (clAr[2] == color[2])):
            return True
    else:
        for i in clAr:
            if ((i[0] == color[0]) and (i[1] == color[1]) and (i[2] == color[2])):
                return True
    return False

def getMinMax(nested_array):
    array = nested_array[0][0]
    min_x = array [0][0]
    min_y = array[0][1]
    max_x = array[0][0]
    max_y = array[0][1]
    for i in range(len(array)):
        #print(array[i][0])
        #print(array[i][1])
        if array[i][0] < min_x:
            min_x = array[i][0]
        if array[i][0] > max_x:
            max_x = array[i][0]

        if array[i][1] < min_y:
            min_y = array[i][1]
        if array[i][1] > max_y:
            max_y = array[i][1]
    return [min_x, max_x, min_y, max_y]

def colorNumber(value, boundaries):
    #print(boundaries)
    for i in range(len(boundaries)):
        if i == (len(boundaries) - 1):
            return i
        if value < boundaries[i + 1]:
            return i

def getShapes(value, contour, colorBar):
    cl_nm = colorNumber(value, colorBar.boundaries)
    return getContourVerts(contour, cl_nm)

def arrayConvert(shapeArray):
    array = shapeArray[0][0]
    poly_array = []
    for i in array:
        x = i[0]
        y = i[1]
        poly_array.append((x,y))
    return poly_array

def meanCentroid(shape_array1, shape_array2, shape_array3 = None):
    shape1 = arrayConvert(shape_array1)
    shape2 = arrayConvert(shape_array2)
    poly1 = Polygon(shape1)
    poly2 = Polygon(shape2)
    center1 = poly1.centroid.coords
    center2 = poly2.centroid.coords
    if not (shape_array3 == None):
        shape3 = arrayConvert(shape_array3)
        poly3 = Polygon(shape3)
        center3 = poly3.centroid.coords
        mean_x = (center1[0][0] + center2[0][0] + center3[0][0])/3.
        mean_y = (center1[0][1] + center2[0][1] + center3[0][1])/3.
        return [mean_x, mean_y]
    mean_x = (center1[0][0] + center2[0][0])/2.
    mean_y = (center1[0][1] + center2[0][1])/2.
    return [mean_x, mean_y]

def modelLambdaRadius(B_array, AngleCol_array, AngleCross_array, lmbda, radius):
    cntr_array = twoDimContour(31, 31)
    #Col
    CS = cntr_array[0]
    CB = cntr_array[1]
    #Cross
    ES = cntr_array[2]
    EB = cntr_array[3]
    #Mag
    DS = cntr_array[4]
    DB = cntr_array[5]

    #coord arrays
    coord_x = []
    coord_y = []

    for i in range(len(B_array)):
        col = AngleCol_array[i]
        cross = AngleCross_array[i]
        bmag = B_array[i]
        #print(str(col) + ', ' + str(cross) + ', ' + str(bmag))
        coord = meanCentroid(getShapes(col, CS, CB), getShapes(cross, ES, EB), getShapes(bmag, DS, DB))
        coord_x.append(coord[0])
        coord_y.append(coord[1])
        #print(coord)

    percent = percentError(coord_x, coord_y, lmbda, radius)
    print('Percent Error for lambda: ' + str(percent[0]))
    print('Percent Error for radius: ' + str(percent[1]))

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
    probes = np.array([[rad*L, lmda*9 , 0], [0, lmda*9 , rad*L]])

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

            B = biot_savart(probes[0], I, wr.p, delta = 0.1)
            B1 = biot_savart(probes[1], I, wr.p, delta = 0.1)
            B_L.append(mag(B))
            B_L1.append(mag(B1))
            angleCol_L.append(angleCol(B, probes[0]))
            angleCol_L1.append(angleCol(B1, probes[1]))
            angleCross_L.append(angleCrossSection(B, probes[0]))
            angleCross_L1.append(angleCrossSection(B1, probes[1]))

        lambda_array.append(lmda/j) # Produce gird array for the helix's lambda
        B_array.append(B_L)
        angleCol_array.append(angleCol_L)
        angleCross_array.append(angleCross_L)
        B_array1.append(B_L1)
        angleCol_array1.append(angleCol_L1)
        angleCross_array1.append(angleCross_L1)

    #print(B_array)
    #print(angleCol_array)
    #print(angleCross_array)

    # Plot different contour maps
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    #ax1 = fig.add_subplot(122)
    CS = ax.contourf(lambda_array, radius_array, angleCol_array, ncl)
    #CS1 = ax1.contourf(lambda_array, radius_array, angleCol_array1, ncl)
    CB = fig.colorbar(CS, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Angle of Magnetic Field Vector Relative to the Column')

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    #ax1 = fig.add_subplot(122)
    ES = ax.contourf(lambda_array, radius_array, angleCross_array, ncl)
    #ES1 = ax1.contourf(lambda_array, radius_array, angleCross_array1, ncl)
    EB = fig.colorbar(ES, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Angle of Magnetic Field Vector Relative to the Cross Section Plane')

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    #ax1 = fig.add_subplot(122)
    DS = ax.contourf(lambda_array, radius_array, B_array, ncl)
    #DS1 = ax1.contourf(lambda_array, radius_array, B_array1, ncl)
    DB = fig.colorbar(DS, shrink=0.8, extend='both')
    plt.ylabel('Radius of Helix Current [cm]')
    plt.xlabel('Lambda [cm]')
    plt.title('Magnitude of Magnetic Field')

    plt.show()

    return [CS, CB, ES, EB, DS, DB]

def percentError(x_array, y_array, x_actual, y_actual):
    x_avg = np.mean(x_array)
    y_avg = np.mean(y_array)
    percent_x = (x_actual - x_avg)*100/x_actual
    percent_y = (y_actual - y_avg)*100/y_actual
    return [percent_x, percent_y]
#print(meanCentroid(getShapes(-1.4, CS, CB), getShapes(1.0, ES, EB), getShapes(0.03, DS, DB)))


"""HEY MAGNUS, read the comments below:"""
######## Simulate data similar to AHF runs ###########

#Pick a lambda and radius you choose to simualte:
l = 25
r = 3

#Pick how much noise you want for each axis
x_scl = 1.
y_scl = 1.
z_scl = 1.

#Pick number for range of time
t = 100

B_data = timeSeries(l, r, t, x_scl, y_scl, z_scl)

######## Estimate the lamda and radius of helix ######

#This will print out
modelLambdaRadius(B_data[0], B_data[1], B_data[2], l, r)
