from Constants import mu0, pi, Kb,amu,mass_elec,elec

### Dimensional scales
r0 = 0.06 #m
L0 = 3.0 #m
I0 = 1000. #Amps
nden0 = 1e18 #m^-3
rho0 = nden0*amu*40. #kg/m^3
B0 = I0*mu0/(2*pi*r0) #tesla
vA = B0/sqrt(mu0*rho0)
epos0 = array([.11,.14,.17,2.68,2.71,2.74])
tau = L0/vA #s
m = rho0*pi*r0*r0*L0

print("L0 (m)", L0)
print("B0 (T)", B0)
print("tau (s)", tau)
print("vA (m/s)", vA)
print("m (kg)", m)

### Non-dimensional parameters
L = L0/r0
dr = .01
dt = 0.1
I = 1000./I0
rho = 1.
dm = 
