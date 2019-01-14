### Wire simulation, quasi-static approx
import abc

from numpy import array, zeros, cross, gradient as grad

from Utility import getBField
from Wires import Wire,State

class AbstractEngine(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,state,dt):
        self.state = state
        self.dt = dt

    @abc.abstractmethod
    def forceScheme(self):
        """calculate forces from current state"""
        return
    
    @abc.abstractmethod
    def stepScheme(self,forces):
        """calculate new state from forces & current state"""
        return

    def advance(self):
        '''Return new simulation state using '''
        forces = self.forceScheme()
        new_state = self.stepScheme(forces)
        self.state = new_state
        return new_state


class MultiWireEngine(AbstractEngine):
    '''Simulator engine for single wire'''
    delta=.01
    
    def forceScheme(self):
        """Biot-Savart brute force calculation"""
        forces = []
        paths = [wire.p for wire in self.state.items]
        currents = [wire.scalars['I'] for wire in self.state.items]
        for wire in self.state.items:
            if not wire.scalars['is_fixed']:
                B = getBField(wire.p,paths,currents,self.delta)
                dl = grad(wire.p, axis=0)
                F = wire.scalars['I']*cross(dl,B)
            else:
                F = None
            forces.append(F)
        return forces
    
    def stepScheme(self,forces):
        """Forward difference scheme for wires"""
        new_time= self.state.time + self.dt
        new_state = State(self.state.name,time=new_time,items=[])

        for F,wire in zip(forces,self.state.items):
            if F is not None:
                new_p = wire.p + wire.v * self.dt
                new_v = wire.v + F/wire.m * self.dt
                new_m = wire.m.copy()

                # Fix first and final segments
                # TODO: continuity issues at endpoints
                new_v[0,:]  = zeros(3)
                new_v[-1,:] = zeros(3)

                new_wire = Wire(new_p,new_v,new_m,
                            scalars=wire.scalars,
                            fields = wire.fields)

                #new_wire.boundary_conditions(new_time)
                new_wire.interpolate()
            else:
                new_wire = wire
            
            new_state.items.append(new_wire)

        return new_state

    def __repr__(self):
        return "MultiWireEngine\nCurrent state: "+str(self.state)




    


