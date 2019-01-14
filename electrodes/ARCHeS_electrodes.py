### CroFT simulation, quasi-static approx

from Constants import mu0, pi
from numpy import array, zeros, arange, shape, abs, ones, matrix,sqrt,exp
from numpy import diff, mgrid, cos, sin, log, newaxis, linalg, cross,arccos
from numpy import concatenate, arcsin,arctan2,linspace
import mayavi.mlab as mlab
import time

phi0 = -2*pi/6.
blue = (0.34765625,0.5625,0.84375)
copper= (0.84765625,0.5625,0.34375)
grey = (107/256.,109/256.,110/256.)
ceramic = (235./256,239./256,240./256)
def torus(pos,r,R):
    radius = 3.5*.0254 #meters
    dphi, dtheta = pi/25.0, pi/25.0
    [phi,theta] = mgrid[0:2*pi+dphi*.5:dphi,0:2*pi+dtheta*.5:dtheta]
    x =  r*sin(theta) + pos[0]
    y = (R+r*cos(theta))*sin(phi) + pos[1]
    z = (R+r*cos(theta))*cos(phi) + pos[2]
    
    mlab.mesh(x, y, z, color=copper)

def electrode2(phi0):
    radius = 3.5*.0254 #meters
    dr, dtheta = radius/25.0, pi/25.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,pi:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)
    z = 0*theta + .0254/8.
    l,m = shape(x)
    for i in range(0,l):
        for j in range(0,m):
                if y[i,j] > -3./16. *.0254:
                    y[i,j] = -3./16. *.0254
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=copper)
    mlab.mesh(xp, yp, z*0, color=copper)

    xmax = (radius**2. - (.0254*3./16)**2.)**.5

    dz = .0254/8.
    dx = xmax
    [x,z] = mgrid[-xmax:xmax+dx*.5:dx,0:.0254/8.+.5*dz:dz]
    y = x*0 - .0254*3./16.
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=copper)

    phi = arcsin(.0254*3./16./radius)
    [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,pi+phi:2*pi-phi +dtheta*.5:dtheta]
    x = radius*cos(theta+phi0)
    y = radius*sin(theta+phi0)
    mlab.mesh(x, y, z, color=copper)

    radius = .323*.0254 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)-2.*.0254
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    z = 0*theta + .0254/8.+.0001
    mlab.mesh(xp, yp, z, color=(0,0,0))

    y0=-2.*.0254
    x = -y0*sin(phi0)
    y = y0*cos(phi0)
    return array([x,y])

def electrode3():
    radius = 3.5*.0254 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) + 9.75*.0254
    z = 0*theta + .0254/8.
    mlab.mesh(x, y, z, color=blue)
    mlab.mesh(x, y, z*0, color=blue)

    dz = .0254/8.
    [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) + 9.75*.0254
    mlab.mesh(x, y, z, color=blue)

    radius = .323*.0254 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) + 9.75*.0254
    z = 0*theta + .0254/8.+.0001
    mlab.mesh(x, y, z, color=(0,0,0))

    return array([0,9.75*.0254])

def electrode4():
    radius = 3.5*.0254 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) - 9.75*.0254
    z = 0*theta + .0254/8.
    mlab.mesh(x, y, z, color=copper)
    mlab.mesh(x, y, z*0, color=copper)

    dz = .0254/8.
    [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) - 9.75*.0254
    mlab.mesh(x, y, z, color=copper)

    radius = .323*.0254 #meters
    dr, dtheta = radius/10.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) - 9.75*.0254
    z = 0*theta + .0254/8.+.0001
    mlab.mesh(x, y, z, color=(0,0,0))

    return array([0,-9.75*.0254])


def upper_candelabra_arm():
    radius = 1.5*.0254
    d = 4.15*.0254
    dz, dtheta = d/2., pi/50.0
    [z,theta] = mgrid[-d:0+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) + 9.75*.0254
    mlab.mesh(x, y, z, color=ceramic)

    r1 = 2.07*.0254 #meters
    r2 = 1.5*.0254
    dr, dtheta = (r2-r1)/5.0, pi/10.0
    [r,theta] = mgrid[r1:r2+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) + 9.75*.0254
    z = 0*theta -d
    mlab.mesh(x, y, z, color=grey)

    radius = 2.07*.0254
    d1 = d +10.5*.0254 + radius
    dz, dtheta = (d1-d)/5.0, pi/50.0
    [z,theta] = mgrid[-d1:-d+dz*.2:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) + 9.75*.0254
    mlab.mesh(x, y, z, color=grey)

def center_candelabra_arm(phi0):
    radius = 1.5*.0254
    d = 4.15*.0254
    dz, dtheta = d/5.0, pi/25.0
    [z,theta] = mgrid[-d:0+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) + 2*.0254

    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=ceramic)

    [z,theta] = mgrid[-d:0+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) - 2*.0254

    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=ceramic)

    radius = 4.08*.0254
    dr, dtheta = radius/5.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)
    z = 0*theta -d
    mlab.mesh(x, y, z, color=grey)

    radius = 4.08*.0254
    d1 = d +8.25*.0254
    dz, dtheta = (d1-d)/5.0, pi/50.0
    [z,theta] = mgrid[-d1:-d+dz*.2:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta)
    mlab.mesh(x, y, z, color=grey)

    r1 = 2*.0254 #meters
    r2 = 4.08*.0254
    dr, dtheta = (r2-r1)/5.0, pi/10.0
    [r,theta] = mgrid[r1:r2+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)
    z = 0*theta -d1
    mlab.mesh(x, y, z, color=grey)

def lower_candelabra_arm():
    radius = 1.5*.0254
    d = 4.15*.0254
    dz, dtheta = d/5.0, pi/50.0
    [z,theta] = mgrid[-d:0+.5*dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) - 9.75*.0254
    mlab.mesh(x, y, z, color=ceramic)

    r1 = 2.07*.0254 #meters
    r2 = 1.5*.0254
    dr, dtheta = (r2-r1)/5.0, pi/10.0
    [r,theta] = mgrid[r1:r2+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta) - 9.75*.0254
    z = 0*theta -d
    mlab.mesh(x, y, z, color=grey)

    radius = 2.07*.0254
    d1 = d +10.5*.0254 + radius
    dz, dtheta = (d1-d)/5.0, pi/50.0
    [z,theta] = mgrid[-d1:-d+dz*.2:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta) - 9.75*.0254
    mlab.mesh(x, y, z, color=grey)

def candelabra_base():
    radius = 2.07*.0254
    d1 = (4.15 +8.25)*.0254
    d2 = d1+13.5*.0254
    dz, dtheta = (d2-d1)/5.0, pi/50.0
    [z,theta] = mgrid[-d2:-d1+dz*.2:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    y = radius*sin(theta)
    mlab.mesh(x, y, z, color=grey)

    radius = 2.07*.0254
    d1 = 4.14*.0254 +(10.5)*.0254 + radius
    l = .0254*(19.5+4.14)/2.
    dy, dtheta = (19.5+4.14)*.0254/25.0, pi/50.0
    [y,theta] = mgrid[-l:l+dy*.1:dy,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)
    z = radius*sin(theta)- d1
    mlab.mesh(x, y, z, color=grey)

    radius = 2.07*.0254
    d1 = 4.14*.0254 +(10.5)*.0254 + radius
    dr, dtheta = radius/5.0, pi/10.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    z = r*sin(theta) - d1
    y = 0*theta -l
    mlab.mesh(x, y, z, color=grey)
    mlab.mesh(x, -y, z, color=grey)

def get_stuffing_coils(a1,a2,b1,b2,plot=False):
    coil_a1 = get_stuff_coil(dx=a1[0], dy=a1[1])
    coil_a2 = get_stuff_coil(dx=a2[0], dy=a2[1])
    coil_b1 = get_stuff_coil(dx=b1[0], dy=b1[1])
    coil_b2 = get_stuff_coil(dx=b2[0], dy=b2[1])
    return [coil_a1,coil_a2,coil_b1,coil_b2]  

def get_stuff_coil(dx=0, dy=0, dz=0, r=.02, d=.10, nturns=20.):#n=208
    theta = arange(0,2*pi*nturns,pi/10.)
    path = array([r*cos(theta)+dx,r*sin(theta)+dy,-d*theta/(2*pi*nturns)+dz]).T
    return path
    
def show_plasma(posA,posB):
    loop_A_lower_pos,loop_A_upper_pos =posA
    loop_B_lower_pos,loop_B_upper_pos =posB
    # Initial loop positions
    theta = arange(0, pi, pi /(150.))
    # Initial loop shape
    a0 = .004
    R_0= .5*linalg.norm(loop_A_upper_pos - loop_A_lower_pos)/1.3
    s = cos(theta)
    a= 0*theta+a0*abs(a0*sin(theta)/2. + 1)
    
    p0,p1 = loop_A_upper_pos, loop_A_lower_pos
    one = matrix(ones((1,len(s))) ).T
    p = matrix(s).T*matrix((p1-p0)/2.) + one*matrix((p1+p0)/2.)
    path1 = array([p[:,0].A1, p[:,1].A1, R_0*sin(theta)]) # Circular
    
    p0,p1 = loop_B_upper_pos, loop_B_lower_pos
    p = matrix(s).T*matrix((p1-p0)/2.) + one*matrix((p1+p0)/2.)
    path2 = array([p[:,0].A1, p[:,1].A1, R_0*sin(theta)]) # Circular
    
    paths = [path1,path2]
    for i in range(0,len(paths)):
        line = mlab.pipeline.line_source(paths[i][0], paths[i][1], paths[i][2], a)
        tube = mlab.pipeline.tube(line, tube_radius=a[0], tube_sides=10)
        tube.filter.vary_radius = 'vary_radius_by_scalar'
        mlab.pipeline.surface(tube, color=(1,0,0))

def plot_all(plot_candelabra=True,plot_loops=False):
    '''Plot electrodes'''

    # Plot electrodes
    loop_B_upper_pos = electrode1(phi0)
    loop_A_lower_pos = electrode2(phi0)
    loop_A_upper_pos = electrode3()
    loop_B_lower_pos = electrode4()

    if plot_candelabra:
        upper_candelabra_arm()
        lower_candelabra_arm()
        center_candelabra_arm(phi0)
        candelabra_base()

    if plot_loops:
        show_plasma([loop_A_lower_pos,loop_A_upper_pos],[loop_B_lower_pos,loop_B_upper_pos])
    return [loop_A_lower_pos,loop_A_upper_pos],[loop_B_lower_pos,loop_B_upper_pos]


if __name__ == "__main__":
    for xpos in [.11,.14,.17,2.68,2.71,2.74]:
        torus([xpos,0,0],.0075,.1)
    mlab.show()
