import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)

# Parameters and constants
R = 1.0e-6          # Particle radius (1 micrometer) [m]
eta = 1.0e-3        # Water viscosity [Pa*s]
T_ref = 300.0           # Temperature [K]
k_B = 1.3806e-23    # Boltzmann constant [J/K]

# Derived constants
gamma = 6 * np.pi * eta * R  # Friction coefficient [N*s/m], given by Stokes formula
D_t = k_B * T_ref / gamma        # Translational diffusion coefficient [m^2/s], given by Einstein relation

# Simulation parameters
dt = 1.0e-5         # Time step [s]
num_steps = 50000   # Total number of steps (0.5 s)
save_interval = 50  # Save every 'save_interval' steps

# Initialization
r = np.array([0.0, 0.0])  # Initial position [m] -> (x, y)
positions = []
times = []

# Main loop
for step in range(num_steps):
    random_displacement = np.sqrt(2 * D_t * dt) * np.random.normal(size=2)

    r = r + random_displacement

    if step % save_interval == 0:
        positions.append(r.copy())
        times.append(step * dt)


positions = np.array(positions)      # shape (N, 2)
times = np.array(times)

# Convert positions to micrometers (um) for plots
positions_um = positions * 1e6
times_s = times

# Plotting
plt.figure(figsize=(7, 7))
plt.plot(positions_um[:, 0], positions_um[:, 1], lw=0.5, alpha=0.8)
plt.plot(positions_um[0, 0], positions_um[0, 1], 'o', color='black', markersize=8, label='Start')
plt.plot(positions_um[-1, 0], positions_um[-1, 1], 'x', color='red', markersize=8, label='End')
plt.xlabel(r'x position ($\mu$m)')
plt.ylabel(r'y position ($\mu$m)')
plt.title('Trajectory of Free Brownian Particle')
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("brownian.png")
plt.show()
