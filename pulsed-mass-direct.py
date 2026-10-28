import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Load thermophoretic mobility data.
T_data_C, DTplot = np.loadtxt(
    "braibanti-fig3.csv", delimiter=",", skiprows=1
).T
T_data_K = T_data_C + 273.15
DT_SI = DTplot * 1e-12

print(DT_SI)
print(T_data_K)

# Temperature measurements from Alireza hoca.
d = np.array([3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 5.0])
T_C = np.array([48, 41, 35, 31, 28, 27, 26, 25])
T = T_C + 273.15


def model(x, A, k, C):
    """Temperature model: T(x) = A * exp(-k*x) + C."""
    return A * np.exp(-k * x) + C


A0 = T[0] - T[-1]
k0 = 1.0
C0 = T[-1]

popt, pcov = curve_fit(model, d, T, p0=[A0, k0, C0])
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

# ---------------------- Physical parameters ----------------------

np.random.seed(24)

R = 2.0e-6
eta = 1.0e-3
T0 = 298.0
k_B = 1.3806e-23

gamma = 6 * np.pi * eta * R
D_brown = k_B * T0 / gamma

# Particle mass for inertial update.
m = 1.0e-13

# Preserve the notebook's fixed temperature-fit parameters.
A_fit = 47487805.196155205
k_fit = 4.262380207691129
C_fit = 297.5165761061297

# ---------------------- Pulse schedule ----------------------

period = 100e-6
pulse_start = 0.0
pulse_width = 10e-9

# Cooling time constant: amplitude decays toward zero.
tau_cool = 100e-6

# ---------------------- Time steps ----------------------

dt_on = 1e-9
dt_off = 1e-6

total_time = 5.0
num_periods = int(np.round(total_time / period))

n_on = int(np.round(pulse_width / dt_on))
n_post = int(np.round((period - pulse_width) / dt_off))

save_every_periods = 10

# ---------------------- Initial position ----------------------

r0_um = 3.0
r = np.array([r0_um, 0.0]) * 1e-6
positions = [r.copy()]

# Position history; initially zero velocity.
r_im1 = r.copy()
r_im2 = r.copy()

# ---------------------- Functions ----------------------


def D_T(T_loc):
    """Temperature-dependent thermophoretic mobility."""
    T_loc = np.clip(T_loc, T_data_K[0], T_data_K[-1])
    return np.interp(T_loc, T_data_K, DT_SI)


def temperature_and_grad(r_xy, A_now):
    x, y = r_xy
    rr = np.sqrt(x**2 + y**2)
    rr_um = rr * 1e6

    T_loc = A_now * np.exp(-k_fit * rr_um) + C_fit

    # Convert the radial derivative from per micrometer to per meter.
    dT_dr = (
        -A_now * k_fit * np.exp(-k_fit * rr_um)
    ) * 1e6

    eps = 1e-30
    ux = x / (rr + eps)
    uy = y / (rr + eps)

    gradT = dT_dr * np.array([ux, uy])
    return T_loc, gradT


def decay_amp(A_now, dt):
    return A_now * np.exp(-dt / tau_cool)


def inertial_position_update(x_im1, x_im2, dt, T_loc):
    """
    Implement the notebook's Eq. (6) update component-wise in 2D.

    x_im1: previous position
    x_im2: position two steps earlier
    """
    denom = 1.0 + dt * (gamma / m)
    a = (2.0 + dt * (gamma / m)) / denom
    b = -1.0 / denom

    noise_coeff = (
        np.sqrt(2.0 * k_B * T_loc * gamma)
        / (m * denom)
    ) * dt**1.5

    w = np.random.normal(size=2)
    return a * x_im1 + b * x_im2 + noise_coeff * w


# ---------------------- Simulation ----------------------

A_t = 0.0

for p in range(num_periods):
    # Laser ON.
    A_t = A_fit

    for i in range(n_on):
        T_loc, gradT = temperature_and_grad(r, A_t)
        drift = -D_T(T_loc) * gradT * dt_on

        r_free = inertial_position_update(
            r_im1, r_im2, dt_on, T_loc
        )
        r = r_free + drift

        r_im2 = r_im1
        r_im1 = r

    # Laser OFF, with exponentially decaying cooling.
    for i in range(n_post):
        A_t = decay_amp(A_t, dt_off)
        T_loc, gradT = temperature_and_grad(r, A_t)
        drift = -D_T(T_loc) * gradT * dt_off

        r_free = inertial_position_update(
            r_im1, r_im2, dt_off, T_loc
        )
        r = r_free + drift

        r_im2 = r_im1
        r_im1 = r

    if p % save_every_periods == 0:
        positions.append(r.copy())

positions = np.array(positions)
positions_um = positions * 1e6

print("Total simulated time (s):", num_periods * period)
print("Periods:", num_periods)
print("Steps per period:", n_on + n_post)
print("Approx total steps:", num_periods * (n_on + n_post))
print("Saved points:", len(positions_um))

# ---------------------- Plotting ----------------------

plt.figure(figsize=(12, 10))

xmin, xmax = -3.0, 8.0
ymin, ymax = -3.0, 3.0
Nx = Ny = 320

xx = np.linspace(xmin, xmax, Nx) * 1e-6
yy = np.linspace(ymin, ymax, Ny) * 1e-6
XX, YY = np.meshgrid(xx, yy)

# Display the laser-ON temperature field.
A_snapshot = A_fit
rr = np.sqrt(XX**2 + YY**2)
TT = A_snapshot * np.exp(-k_fit * rr * 1e6) + C_fit

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

# Uncomment to display the temperature colorbar.
# plt.colorbar(
#     im,
#     label="Temperature (K, log scale)",
#     shrink=0.55,
#     aspect=30,
#     pad=0.02,
# )

print(positions)

plt.plot(
    positions_um[:, 0],
    positions_um[:, 1],
    lw=1.0,
    color="black",
    zorder=2,
    label=fr"trajectory ($m = {m:.1e}\ \mathrm{{kg}}$)",
)
plt.scatter(
    positions_um[0, 0],
    positions_um[0, 1],
    s=70,
    c="green",
    label="start",
    zorder=3,
)
plt.scatter(
    positions_um[-1, 0],
    positions_um[-1, 1],
    s=70,
    c="purple",
    label="end",
    zorder=3,
)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.savefig(
    "pulsed-mass-wrong.png",
    bbox_inches="tight",
    pad_inches=0.2,
)
plt.show()
