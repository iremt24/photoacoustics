Here is `pulsed-original` as one copyable `.py` script. The CSV data is embedded, preserving the original `skiprows=1` behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Embedded braibanti-fig3.csv data.
# The first row is omitted to preserve the original skiprows=1 behavior.
mobility_data = np.array([
    [14.704260651629072, 1.0714285714285703],
    [24.586466165413533, 1.8571428571428568],
    [34.54323308270677, 3.2678571428571432],
    [44.48621553884711, 4.142857142857142],
])

T_data_C, DTplot = mobility_data.T
T_data_K = T_data_C + 273.15
DT_SI = DTplot * 1e-12

print(DT_SI)
print(T_data_K)

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

print("Fit parameters:")
print("A =", A)
print("k =", k)
print("C =", C)

xfit = np.linspace(min(d), max(d), 200)
yfit = model(xfit, *popt)

plt.scatter(d, T, label="data (K)")
plt.plot(xfit, yfit, label="exp fit (K)")
plt.xlabel("d")
plt.ylabel("T (K)")
plt.legend()
plt.show()

np.random.seed(24)

R = 2.0e-6
eta = 1.0e-3
T0 = 298.0
k_B = 1.3806e-23

gamma = 6 * np.pi * eta * R
D_brown = k_B * T0 / gamma

A_fit = 47487805.196155205
k_fit = 4.262380207691129
C_fit = 297.5165761061297

# Pulse schedule: ON then OFF each period.
period = 100e-6
pulse_start = 0.0
pulse_width = 10e-9

# Cooling time constant.
tau_cool = 100e-6

# Time steps.
dt_on = 1e-9
dt_off = 1e-6

total_time = 5.0
num_periods = int(np.round(total_time / period))

n_on = int(np.round(pulse_width / dt_on))
n_post = int(np.round((period - pulse_width) / dt_off))

save_every_periods = 10

# Initial position in meters.
r0_um = 3.0
r = np.array([r0_um, 0.0]) * 1e-6
positions = [r.copy()]


def D_T(T_loc):
    """Temperature-dependent thermophoretic mobility in SI units."""
    T_loc = np.clip(T_loc, T_data_K[0], T_data_K[-1])
    return np.interp(T_loc, T_data_K, DT_SI)


def temperature_and_grad(r_xy, A_now):
    x, y = r_xy
    rr = np.sqrt(x**2 + y**2)
    rr_um = rr * 1e6

    T_loc = A_now * np.exp(-k_fit * rr_um) + C_fit

    # Convert temperature gradient from K/µm to K/m.
    dT_dr = -A_now * k_fit * np.exp(-k_fit * rr_um) * 1e6

    eps = 1e-30
    ux = x / (rr + eps)
    uy = y / (rr + eps)

    gradT = dT_dr * np.array([ux, uy])
    return T_loc, gradT


def decay_amp(A_now, dt):
    return A_now * np.exp(-dt / tau_cool)


sqrt_2D_on = np.sqrt(2.0 * D_brown * dt_on)
sqrt_2D_off = np.sqrt(2.0 * D_brown * dt_off)

# Simulation.
A_t = 0.0

for p in range(num_periods):
    # Laser ON.
    A_t = A_fit

    if n_on > 0:
        noise = np.random.normal(size=(n_on, 2))

        for i in range(n_on):
            T_loc, gradT = temperature_and_grad(r, A_t)
            drift = -D_T(T_loc) * gradT * dt_on
            brown = sqrt_2D_on * noise[i]
            r = r + drift + brown

    # Laser OFF: exponentially decaying cooling.
    if n_post > 0:
        noise = np.random.normal(size=(n_post, 2))

        for i in range(n_post):
            A_t = decay_amp(A_t, dt_off)
            T_loc, gradT = temperature_and_grad(r, A_t)
            drift = -D_T(T_loc) * gradT * dt_off
            brown = sqrt_2D_off * noise[i]
            r = r + drift + brown

    if p % save_every_periods == 0:
        positions.append(r.copy())

positions = np.array(positions)
positions_um = positions * 1e6

print("Total simulated time (s):", num_periods * period)
print("Periods:", num_periods)
print("Steps per period:", n_on + n_post)
print("Approx total steps:", num_periods * (n_on + n_post))
print("Saved points:", len(positions_um))

# Plotting.
plt.figure(figsize=(12, 10))

xmin, xmax = -3.0, 8.0
ymin, ymax = -3.0, 3.0
Nx = Ny = 320

xx = np.linspace(xmin, xmax, Nx) * 1e-6
yy = np.linspace(ymin, ymax, Ny) * 1e-6
XX, YY = np.meshgrid(xx, yy)

# Laser-ON temperature snapshot.
A_snapshot = A_fit
rr = np.sqrt(XX**2 + YY**2)
TT = A_snapshot * np.exp(-k_fit * (rr * 1e6)) + C_fit

TT_safe = np.clip(TT, 1e-6, None)

im = plt.imshow(
    TT_safe,
    extent=[xmin, xmax, ymin, ymax],
    origin="lower",
    cmap="coolwarm",
    norm=LogNorm(vmin=TT_safe.min(), vmax=TT_safe.max()),
    aspect="equal",
    alpha=0.8,
)

plt.colorbar(
    im,
    label="Temperature (K, log scale)",
    shrink=0.55,
    aspect=30,
    pad=0.02,
)

plt.plot(
    positions_um[:, 0],
    positions_um[:, 1],
    lw=1.0,
    color="black",
)

plt.scatter(
    positions_um[0, 0],
    positions_um[0, 1],
    s=70,
    c="green",
    label="start",
)

plt.scatter(
    positions_um[-1, 0],
    positions_um[-1, 1],
    s=70,
    c="purple",
    label="end",
)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.savefig("pulsed-laser.png")
plt.show()
```
