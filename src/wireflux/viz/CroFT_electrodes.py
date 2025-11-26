### CroFT simulation, quasi-static approx

from wireflux.utils.constants import mu0, pi
from numpy import array, zeros, arange, shape, abs, ones, matrix,sqrt,exp
from numpy import diff, mgrid, cos, sin, log, newaxis, linalg, cross,arccos
from numpy import concatenate, arcsin,arctan2,linspace
import mayavi.mlab as mlab
import time
import numpy as np
import pyvista as pv

phi0 = -2*pi/6.
blue = (0.34765625,0.5625,0.84375)
copper= (0.84765625,0.5625,0.34375)
grey = (107/256.,109/256.,110/256.)
ceramic = (235./256,239./256,240./256)

def _electrode1(phi0):
    radius = 3.5*.0254 #meters
    dr, dtheta = radius/25.0, pi/25.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)
    z = 0*theta + .0254/8.
    l,m = shape(x)
    for i in range(0,l):
        for j in range(0,m):
                if y[i,j] < 3./16. *.0254:
                    y[i,j] = 3./16. *.0254
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=blue)
    mlab.mesh(xp, yp, z*0, color=blue)

    xmax = (radius**2. - (.0254*3./16)**2.)**.5

    dz = .0254/8.
    dx = xmax
    [x,z] = mgrid[-xmax:xmax+dx*.5:dx,0:.0254/8.+.5*dz:dz]
    y = x*0 + .0254*3./16.
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    mlab.mesh(xp, yp, z, color=blue)

    phi = arcsin(.0254*3./16./radius)
    [z,theta] = mgrid[0:.0254/8.+.5*dz:dz,phi:pi-phi +dtheta*.5:dtheta]
    x = radius*cos(theta+phi0)
    y = radius*sin(theta+phi0)
    mlab.mesh(x, y, z, color=blue)

    radius = .323*.0254 #meters
    dr, dtheta = radius/250.0, pi/250.0
    [r,theta] = mgrid[0:radius+dr*.5:dr,0:2*pi+dtheta*.5:dtheta]
    x = r*cos(theta)
    y = r*sin(theta)+2.*.0254
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)
    z = 0*theta + .0254/8.+.0001
    mlab.mesh(xp, yp, z, color=(0,0,0))

    y0=2.*.0254
    x = -y0*sin(phi0)
    y = y0*cos(phi0)
    return array([x,y])

def make_structured_mesh(x, y, z):
    """
    Convert 2D x,y,z arrays (m,n) into a PyVista StructuredGrid.
    VTK expects 3D dims, so we append a dummy depth of 1.
    """
    m, n = x.shape
    grid = pv.StructuredGrid()

    # PyVista needs points in (N, 3) shape
    pts = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)]
    grid.points = pts

    # KEY FIX: structured grid dimensions must be (m, n, 1)
    grid.dimensions = (m, n, 1)

    return grid


def electrode1(phi0, plotter=None):
    import numpy as np
    import pyvista as pv
    from numpy import cos, sin, pi, mgrid, arcsin, shape

    blue = (0.34765625, 0.5625, 0.84375)

    if plotter is None:
        plotter = pv.Plotter()
        plotter.set_background("white")

    # ---------------------------------------------------------
    # 1) MAIN D-SHAPED CAP (curved surface + clipped inner flat)
    # ---------------------------------------------------------
    radius = 3.5 * 0.0254
    dr = radius / 25.0
    dtheta = pi / 25.0

    r, theta = mgrid[0:radius + 0.5*dr:dr, 0:pi + 0.5*dtheta:dtheta]
    x = r * cos(theta)
    y = r * sin(theta)
    z = np.zeros_like(theta) + (0.0254/8.0)

    # clip inside flat region
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            if y[i, j] < (3.0/16.0)*0.0254:
                y[i, j] = (3.0/16.0)*0.0254

    # rotate by phi0
    xp = x*cos(phi0) - y*sin(phi0)
    yp = x*sin(phi0) + y*cos(phi0)

    grid1 = make_structured_mesh(xp, yp, z)
    plotter.add_mesh(grid1, color=blue)

    # also add the flat bottom cap (z = 0)
    grid1b = make_structured_mesh(xp, yp, np.zeros_like(z))
    plotter.add_mesh(grid1b, color=blue)

    # ---------------------------------------------------------
    # 2) FLAT BOTTOM RECTANGULAR STRIP
    # ---------------------------------------------------------
    xmax = (radius**2 - (0.0254*3.0/16.0)**2)**0.5

    dz = 0.0254/8.0
    dx = xmax

    x2, z2 = mgrid[-xmax:xmax+0.5*dx:dx, 0:dz+0.5*dz:dz]
    y2 = x2*0 + (3.0/16.0)*0.0254

    xp2 = x2*cos(phi0) - y2*sin(phi0)
    yp2 = x2*sin(phi0) + y2*cos(phi0)

    grid2 = make_structured_mesh(xp2, yp2, z2)
    plotter.add_mesh(grid2, color=blue)

    # ---------------------------------------------------------
    # 3) SIDE RADIAL ARC SURFACE
    # ---------------------------------------------------------
    phi = arcsin((3.0/16.0)*0.0254 / radius)
    z3, theta3 = mgrid[0:dz+0.5*dz:dz, phi:pi-phi+0.5*dtheta:dtheta]

    x3 = radius * cos(theta3 + phi0)
    y3 = radius * sin(theta3 + phi0)
    z3 = z3 + 0.0   # ensure same shape

    grid3 = make_structured_mesh(x3, y3, z3)
    plotter.add_mesh(grid3, color=blue)

    # ---------------------------------------------------------
    # 4) SMALL CYLINDER CAP OFFSET IN +Y DIRECTION
    # ---------------------------------------------------------
    radius2 = 0.323 * 0.0254
    dr2 = radius2 / 250.0
    dtheta2 = 2*pi / 250.0

    r4, theta4 = mgrid[0:radius2+0.5*dr2:dr2, 0:2*pi+0.5*dtheta2:dtheta2]
    x4 = r4 * cos(theta4)
    y4 = r4 * sin(theta4) + (2.0 * 0.0254)

    # rotate
    xp4 = x4*cos(phi0) - y4*sin(phi0)
    yp4 = x4*sin(phi0) + y4*cos(phi0)

    z4 = np.zeros_like(theta4) + (0.0254/8.0) + 0.0001

    grid4 = make_structured_mesh(xp4, yp4, z4)
    plotter.add_mesh(grid4, color=(0,0,0))

    # ---------------------------------------------------------
    # Return the offset pivot point (matching your old code)
    # ---------------------------------------------------------
    y0 = 2.0 * 0.0254
    x_final = -y0 * sin(phi0)
    y_final =  y0 * cos(phi0)

    return np.array([x_final, y_final]), plotter



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

######################
### Coil Parameters###
######################
theta = arange(0,20*pi,pi/100.)
coil_R = .04445 #coil major radius
ypos = .0879    #m above electrodes
z_pitch = .01016# z travel per loop
dx = 0 #x offset of coils

#left coil
left_coil = array([z_pitch*theta/(2.*pi) + .1049, coil_R*cos(theta)+dx,coil_R*sin(theta) + ypos, ]).T
#right coil
right_coil = array([-z_pitch*theta/(2.*pi) - .1049, coil_R*cos(theta)+dx,-coil_R*sin(theta) + ypos, ]).T


if __name__ == "__main__":
    a,plotter = electrode1(phi0=-2*np.pi/6)
    plotter.show()

    ##lower electrode +
    b1 = electrode2(phi0)
    path = get_stuff_coil(dx=b1[0], dy=b1[1],r= .02, d=.08, nturns=12)
    mlab.plot3d(path[:,0], path[:,1], path[:,2], tube_radius=.002,
                        color=copper)
    radius = 1.4*.0254
    d = .04
    dz, dtheta = (d)/5.0, pi/50.0
    [z,theta] = mgrid[-d:0+dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta) + b1[0]
    y = radius*sin(theta) + b1[1]
    mlab.mesh(x, y, z, color=copper)
    
    radius = .003
    d = 4.15*.0254
    dz, dtheta = (d)/5.0, pi/50.0
    [z,theta] = mgrid[-d:0+dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta)+ b1[0]
    y = radius*sin(theta)+ b1[1]
    mlab.mesh(x, y, z, color=grey)
    
    ##B-field background
    m=-1;x0=b1[0];y0=b1[1]
    X, Y, Z = mgrid[-24:24,-24:24,0:45]/200. # meters
    rho = sqrt((X-x0)**2 + (Y-y0)**2)
    
    r1 = sqrt(rho**2 + Z**2) 
    Bx = 3*(X-x0)* m*Z/(r1**5)
    By = 3*(Y-y0)* m*Z/(r1**5)
    Bz = 3*Z* m*Z/(r1**5) - m/(r1**3)
    
    ##density cone
    cones = ((.01/(.01+Z))**2 )*exp(-1.15*rho*rho/((.01+Z)**2))
    
    ##Upper electrode
    b0 = electrode1(phi0)
    path = get_stuff_coil(dx=b0[0], dy=b0[1],r= .02, d=.08, nturns=12)
    mlab.plot3d(path[:,0], path[:,1], path[:,2], tube_radius=.002,
                        color=blue)
    radius = 1.4*.0254
    d = .04
    dz, dtheta = (d)/5.0, pi/50.0
    [z,theta] = mgrid[-d:0+dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta) + b0[0]
    y = radius*sin(theta) + b0[1]
    mlab.mesh(x, y, z, color=blue)
    
    radius = .003
    d = 4.15*.0254
    dz, dtheta = (d)/5.0, pi/50.0
    [z,theta] = mgrid[-d:0+dz:dz,0:2*pi+dtheta*.5:dtheta]
    x = radius*cos(theta) + b0[0]
    y = radius*sin(theta) + b0[1]
    mlab.mesh(x, y, z, color=grey)
    
    ##B-field background
    m=1;x0=b0[0];y0=b0[1]
    rho = sqrt((X-x0)**2 + (Y-y0)**2)
    
    r1 = sqrt(rho**2 + Z**2) 
    Bx += 3*(X-x0)* m*Z/(r1**5)
    By += 3*(Y-y0)* m*Z/(r1**5)
    Bz += 3*Z* m*Z/(r1**5) - m/(r1**3)
    
    Bsrc = mlab.pipeline.vector_field(X,Y,Z,Bx,By,Bz)
    Bmag = mlab.pipeline.extract_vector_norm(Bsrc)
    
    ##density cone
    cones += ((.01/(.01+Z))**2 )*exp(-1.15*rho*rho/((.01+Z)**2))
    
    ##plot fieldlines    
    flow = mlab.pipeline.streamline(Bmag, seedtype='plane',
                                    seed_visible=True,
                                    seed_resolution=3,
                                    integration_direction = 'both')
    ##plot density cones         
    src = mlab.pipeline.scalar_field(X,Y,Z,log(cones+.01))
    mlab.pipeline.iso_surface(src, contours=[-4.36,-4.2,-4,-3.5,-3,-2,-1,-.5], opacity=0.2)
    #mlab.pipeline.iso_surface(src, contours=[cones.max()*.01], opacity=0.5)
    #iso = mlab.pipeline.iso_surface(X,Y,Z,cones)

    ### Circle path
    n=100;theta = linspace(0,pi,n)
    r0 = .01 #meters
    start,end = array(b1),array(b0)
    pvec = end-start
    st = zeros(3)
    st[0:2] = end
    R0 = linalg.norm(pvec)/2.
    y,z = R0*cos(theta),R0*sin(theta)
    yaxis,zaxis = zeros((3,1)),zeros((3,1))
    yaxis[1,0],zaxis[2,0] = pvec[1]/(R0*2), 1.
    yaxis[0,0] = pvec[0]/(R0*2)

    ppath = ((y-y[0])*yaxis + z*zaxis).T + st

    mlab.plot3d(ppath[:,0], ppath[:,1], ppath[:,2], tube_radius=r0, color=(1,0,0))
    mlab.show()
