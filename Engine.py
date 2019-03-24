### Wire simulation, quasi-static approx
import abc

from numpy import array, zeros, cross, gradient as grad,linalg,pi

from Utility import getBField, get_R, get_normal
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


class SingleWireEngine(AbstractEngine):
    '''Simulator engine for single wire'''
    delta=.01

    def JxB_force(self,wire,B):
        '''
        Given a wire path, a current I and magnetic vector B at each point,
        calculates the JxB force at each point
        '''
        dl = grad(wire.p, axis=0)
        return wire.I*cross(dl,B)

    def tension_force(self,wire):
        '''
        Calculates tension force from 3D curve properties
        '''
        T,CumLen,dl,N,R = wire.get_3D_curve_params()
        vol = pi*wire.r*wire.r*dl
        Lsq = CumLen[-1]**2
        ft = vol*(wire.Bp*wire.Bp/R)*((Lsq - wire.L_init**2)/Lsq)*N.T
        return ft.T
    
    def forceScheme(self):
        """Force calculation using Biot-Savart B-field calc"""
        forces = []
        wire = self.state.items[0]
        path = wire.p
        current = wire.I
        B = getBField(wire.p,[path],[current],self.delta)
        
        JxB = self.JxB_force(wire,B)
        tension = self.tension_force(wire)
        #print(tension/JxB)
        
        F = JxB + tension
        F[0:2,:] = 0
        F[-2:,:] = 0
        
        return F        
    
    def stepScheme(self,forces):
        """Forward difference scheme for wires"""
        new_time= self.state.time + self.dt
        new_state = State(self.state.name,time=new_time,items=[])

        for F,wire in zip(forces,self.state.items):
            if F is not None:
                new_p = wire.p + wire.v * self.dt
                new_v = wire.v + F/wire.m* self.dt
                new_m = wire.m.copy()

                print(linalg.norm(new_v,axis=1).max()*self.dt)

                ### Boundary conditions
                # Fix first and final segments
                new_v[0:2,:]= 0.
                new_v[-2:,:]= 0.

                # impervious lower boundary
                r0=0.1
                new_v[2:-2,2][new_p[2:-2,2] < r0] = 0
                new_p[2:-2,2][new_p[2:-2,2] < r0] = r0

                ### Initialize new wire 
                new_wire = Wire(new_p,new_v,new_m,wire.I,r=wire.r,Bp=wire.Bp)
                new_wire.interpolate()
            else:
                new_wire = wire
            
            new_state.items.append(new_wire)

        return new_state

    def __repr__(self):
        return "SingleWireEngine\nCurrent state: "+str(self.state)



class MultiWireEngine(AbstractEngine):
    '''Simulator engine for multiple wires'''
    delta=.01

    def JxB_force(self,wire,B):
        '''
        Given a wire path, a current I and magnetic vector B at each point,
        calculates the JxB force at each point
        '''
        dl = grad(wire.p, axis=0)
        return wire.I*cross(dl,B)

    def tension_force(self,wire):
        '''
        Calculates tension force from 3D curve properties
        '''
        T,CumLen,dl,N,R = wire.get_3D_curve_params()
        vol = pi*wire.r*wire.r*dl
        Lsq = CumLen[-1]**2
        ft = vol*(wire.Bp*wire.Bp/R)*((Lsq - wire.L_init**2)/Lsq)*N.T
        return ft.T
    
    def forceScheme(self):
        """Force calculation using Biot-Savart B-field calc"""
        forces = []
        paths = [wire.p for wire in self.state.items]
        currents = [wire.I for wire in self.state.items]
        for wire in self.state.items:
            if not wire.is_fixed:
                B = getBField(wire.p,paths,currents,self.delta)
                
                JxB = self.JxB_force(wire,B)
                tension = self.tension_force(wire)
                #print(tension/JxB)
                
                F = JxB + tension
                F[0:2,:] = 0
                F[-2:,:] = 0
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
                new_v = wire.v + F/wire.m* self.dt
                new_m = wire.m.copy()

                print(linalg.norm(new_v,axis=1).max()*self.dt)

                ### Boundary conditions
                # Fix first and final segments
                new_v[0:2,:]= 0.
                new_v[-2:,:]= 0.

                # impervious lower boundary
                r0=0.1
                new_v[2:-2,2][new_p[2:-2,2] < r0] = 0
                new_p[2:-2,2][new_p[2:-2,2] < r0] = r0

                ### Initialize new wire 
                new_wire = Wire(new_p,new_v,new_m,wire.I,r=wire.r,Bp=wire.Bp)
                new_wire.interpolate()
            else:
                new_wire = wire
            
            new_state.items.append(new_wire)

        return new_state

    def __repr__(self):
        return "MultiWireEngine\nCurrent state: "+str(self.state)




    


