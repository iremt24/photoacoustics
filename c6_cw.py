import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

# Load thermophoretic mobility data.
T_data_C, D_th_data_scaled = np.loadtxt(
    'braibanti-fig3.csv', delimiter=',', skiprows=1
).T
T_data_K = T_data_C + 273.15
D_th_data_SI = D_th_data_scaled * 1e-12

print(D_th_data_SI)
print(T_data_K)

# Fit an exponential temperature profile.
d = np.array([3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 5.0])
T_C = np.array([48, 41, 35, 31, 28, 27, 26, 25])
T = T_C + 273.15


def model(x, A, k, C):
    return A * np.exp(-k * x) + C


A0 = T[0] - T[-1]
k0 = 1.0
C0 = T[-1]
p0 = [A0, k0, C0]

popt, pcov = curve_fit(model, d, T, p0=p0)
A, k, C = popt

print('Fit parameters:')
print('A =', A)
print('k =', k)
print('C =', C)

xfit = np.linspace(min(d), max(d), 200)
yfit = model(xfit, *popt)

plt.scatter(d, T, label='data (K)')
plt.plot(xfit, yfit, label='exp fit (K)')
plt.xlabel('d')
plt.ylabel('T (K)')
plt.legend()
plt.savefig('exp-fit.png')
plt.show()

# Simulation parameters.
np.random.seed(24)

R = 2.05e-6
eta = 1.0e-3
T_ref = 300.0
k_B = 1.3806e-23

gamma = 6 * np.pi * eta * R
D_t = k_B * T_ref / gamma

# Preserve the fitted constants used in the notebook simulation.
A_fit = 47487805.196155205
k_fit = 4.262380207691129
C_fit = 297.5165761061297

dt = 1e-8
num_steps = 50000000
save_interval = 50

# Start along the +x axis, away from the source at the origin.
r0_um = 3.0
r = np.array([r0_um, 0.0]) * 1e-6
positions = []


def temperature(x, y):
    rr = np.sqrt(x**2 + y**2)
    rr_um = rr * 1e6
    return A_fit * np.exp(-k_fit * rr_um) + C_fit


def gradT(x, y):
    rr = np.sqrt(x**2 + y**2)
    rr_um = rr * 1e6
    eps = 1e-30

    # Temperature derivative with respect to radius in micrometers.
    dT_dr_um = -A_fit * k_fit * np.exp(-k_fit * rr_um)

    # Convert to temperature gradient per meter.
    dT_dr = dT_dr_um * 1e6

    ux = x / (rr + eps)
    uy = y / (rr + eps)

    return dT_dr * np.array([ux, uy])


def D_th(T_local):
    """Temperature-dependent thermophoretic mobility [m^2/(s*K)]."""
    T_local = np.clip(T_local, T_data_K[0], T_data_K[-1])
    return np.interp(T_local, T_data_K, D_th_data_SI)


for step in range(num_steps):
    x, y = r
    T_local = temperature(x, y)
    grad_T_local = gradT(x, y)

    drift = -D_th(T_local) * grad_T_local * dt
    brown = np.sqrt(2 * D_t * dt) * np.random.normal(size=2)
    r = r + drift + brown

    if step % save_interval == 0:
        positions.append(r.copy())

positions = np.array(positions)
positions_um = positions * 1e6

# Plot the trajectory over the temperature field.
plt.figure(figsize=(12, 10))

xmin, xmax = -3.0, 8.0
ymin, ymax = -3.0, 3.0

Nx = Ny = 350
xx = np.linspace(xmin, xmax, Nx) * 1e-6
yy = np.linspace(ymin, ymax, Ny) * 1e-6
XX, YY = np.meshgrid(xx, yy)
TT = temperature(XX, YY)

# LogNorm requires strictly positive values.
TT_safe = np.clip(TT, 1e-6, None)

im = plt.imshow(
    TT_safe,
    extent=[xmin, xmax, ymin, ymax],
    origin='lower',
    cmap='coolwarm',
    norm=LogNorm(vmin=TT_safe.min(), vmax=TT_safe.max()),
    aspect='equal',
    alpha=0.8,
)

plt.colorbar(
    im,
    label='Temperature (K, log scale)',
    shrink=0.55,
    aspect=30,
    pad=0.02,
)

plt.plot(
    positions_um[:, 0],
    positions_um[:, 1],
    lw=1.0,
    color='black',
    zorder=2,
)
plt.scatter(
    positions_um[0, 0],
    positions_um[0, 1],
    s=70,
    c='green',
    label='Start',
    zorder=3,
)
plt.scatter(
    positions_um[-1, 0],
    positions_um[-1, 1],
    s=70,
    c='purple',
    label='End',
    zorder=3,
)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
# plt.savefig('exponential.png')
plt.show()
