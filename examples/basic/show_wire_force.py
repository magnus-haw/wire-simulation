import numpy as np
import matplotlib.pyplot as plt

from wireflux.models.wires import Wire
from wireflux.physics.biot_savart import getBField
from wireflux.physics.forces import JxB_force

# -------------------------------
# Geometry: half-circle wire
# -------------------------------
n = 200
R = 1.0
I = 1.0
rwire = 0.02

theta = np.linspace(0, np.pi, n)
p = np.column_stack((
    R * np.cos(theta),   # x
    np.zeros_like(theta),
    R * np.sin(theta)    # z
))

v = np.zeros_like(p)
m = np.ones((n, 1))

wire = Wire(p, v, m, I, r=rwire, is_fixed=False)

# -------------------------------
# Compute self B-field
# -------------------------------
B = getBField(wire.p, [wire])

# -------------------------------
# Compute J×B force
# -------------------------------
F = JxB_force(wire.p, wire.I, B)

# Absorb endpoint forces (line-tied electrodes)
F[:2, :] = 0.0
F[-2:, :] = 0.0

# -------------------------------
# Project forces
# -------------------------------
# Radial direction in x–z plane
r_vec = np.column_stack((p[:, 0], p[:, 2]))
r_hat = r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)

F_radial = F[:, 0] * r_hat[:, 0] + F[:, 2] * r_hat[:, 1]
F_vertical = F[:, 2]

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(theta, F_radial,'bo-', lw=2)
plt.xlabel(r"$\theta$")
plt.ylabel("Radial force")
plt.title("Expansive (hoop) force")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(theta, F_vertical, 'gs-',lw=2)
plt.xlabel(r"$\theta$")
plt.ylabel("Vertical force")
plt.title("Vertical force component")
plt.grid(True)

plt.tight_layout()
plt.show()
