### Wire simulation, quasi-static approx
import abc
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
from numpy import reshape, shape

from wireflux.physics.biot_savart import getBField
from wireflux.physics.inductance import inductance
from wireflux.physics.forces import JxB_force, pressure_repulsion
from wireflux.utils.smooth import smooth3DVectors
from wireflux.models.wires import Wire
from wireflux.models.newwires import NewWire
from wireflux.core.state import State

def defaultBC(state):
    ### Boundary conditions
    for wire in state.items:
        if not wire.is_fixed:
            
            # Fix first and final segments
            wire.v[0:2,:]= 0.
            wire.v[-2:,:]= 0.

            # impervious lower boundary
            r0=0.1
            wire.v[2:-2,2][wire.p[2:-2,2] < r0] = 0
            wire.p[2:-2,2][wire.p[2:-2,2] < r0] = r0


class AbstractEngine(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,state,dt,bc=defaultBC):
        self.state = state
        self.dt = dt
        self.bc = bc
        self.enable_repulsion = True
        self.repulsion_cutoff = 2.0 * self.state.items[0].r
        self.repulsion_strength = 0.5
        self.repulsion_power = 2

    @abc.abstractmethod
    def forceScheme(self):
        """calculate forces from current state"""
        return
    
    @abc.abstractmethod
    def stepScheme(self,forces):
        """calculate new state from forces & current state"""
        return

    def correctBoundaries(self):
        """implement boundary conditions, corrections"""
        self.bc(self.state)
        return

    def advance(self):
        '''Return new simulation state using '''
        forces = self.forceScheme()
        self.state = self.stepScheme(forces)
        
        self.correctBoundaries()
        return self.state


class MultiWireEngine(AbstractEngine):
    '''Simulator engine for multiple wires'''

    def getB(self,X,Y,Z):
        """Get B-field at arbitrary points using Biot-Savart"""
        paths = [wire.p for wire in self.state.items]
        currents = [wire.I for wire in self.state.items]
        mesh_shape = shape(X)
        
        # init points array in proper shape
        x,y,z = X.ravel(order='C'),Y.ravel(order='C'),Z.ravel(order='C')
        n = len(x)
        points = zeros((n,3))
        points[:,0]=x
        points[:,1]=y
        points[:,2]=z

        # Calculate B-field at each point
        B = getBField(points,self.state.items)

        # Reshape into original mesh matrix
        Bx,By,Bz = B[:,0],B[:,1],B[:,2]
        BXi = reshape(Bx,mesh_shape,order='C')
        BYi = reshape(By,mesh_shape,order='C')
        BZi = reshape(Bz,mesh_shape,order='C')

        return BXi,BYi,BZi
        
    def _forceScheme(self):
        """Force calculation using Biot-Savart B-field calc"""
        forces = []
        paths = [wire.p for wire in self.state.items]
        currents = [wire.I for wire in self.state.items]
        for wire in self.state.items:
            if not wire.is_fixed:
                B = getBField(wire.p,self.state.items)
                JxB = JxB_force(wire.p,wire.I,B)
                
                F = JxB #+ tension_force(wire)

                ###Fixed footpoints
                F[0:2,:] = 0
                F[-2:,:] = 0
                F = smooth3DVectors(F,n=10)
            else:
                F = None
            forces.append(F)
        return forces        
    
    def forceScheme(self):
        forces = []

        # Stack all wire nodes
        offsets = []
        all_points = []
        for wire in self.state.items:
            offsets.append(len(all_points))
            all_points.append(wire.p)
        all_points = np.vstack(all_points)

        # Compute B everywhere once
        B_all = getBField(all_points, self.state.items)

        # Slice per wire
        idx = 0
        for wire in self.state.items:
            n = len(wire.p)
            B = B_all[idx:idx+n]
            idx += n

            if not wire.is_fixed:
                F = JxB_force(wire.p, wire.I, B)
                F[0:2, :] = 0
                F[-2:, :] = 0
                #F = smooth3DVectors(F, n=10)
            else:
                F = None

            forces.append(F)

        # --- Add inter-wire repulsion ---
        if self.enable_repulsion:

            for i in range(len(self.state.items)):
                for j in range(i+1, len(self.state.items)):

                    wi = self.state.items[i]
                    wj = self.state.items[j]

                    if wi.is_fixed and wj.is_fixed:
                        continue

                    F_i, F_j = pressure_repulsion(wi.p, wj.p,1,.5)

                    if forces[i] is not None:
                        forces[i] += F_i
                    if forces[j] is not None:
                        forces[j] += F_j

        # --- Boundary constraints ---
        for k, wire in enumerate(self.state.items):
            if forces[k] is not None:
                forces[k][0:2, :] = 0
                forces[k][-2:, :] = 0
        
        return forces

    def stepScheme(self,forces):
        """Forward difference scheme for wires"""
        new_time= self.state.time + self.dt
        new_state = State(self.state.name,time=new_time,items=[])

        for F,wire in zip(forces,self.state.items):
            if F is not None:
                new_p = wire.p + wire.v * self.dt
                new_v = wire.v + F/wire.m* self.dt
                new_m = wire.m.copy()

                ### Initialize new wire 
                new_wire = NewWire(new_p,new_v,new_m,wire.I,r=wire.r,Bp=wire.Bp,L_init=wire.L_init)
                new_wire.last_force = F
                new_wire.interpolate()
            else:
                new_wire = wire
            new_state.items.append(new_wire)
        return new_state

    def getEnergy(self):
        norm = 1e-7
        energy =0
        for i, wr1 in enumerate(self.state.items):
            for j, wr2 in enumerate(self.state.items[i:], start=i):
                factor = 1.0 if i == j else 2.0
                energy += 0.5 * factor * wr1.I * wr2.I * inductance(...)

        return energy

    def __repr__(self):
        return "MultiWireEngine\nCurrent state: "+str(self.state)

