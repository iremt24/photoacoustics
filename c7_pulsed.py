
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# ---------------------- Thermophoretic mobility data ----------------------
T_data_C, D_th_data_scaled = np.loadtxt(
    'braibanti-fig3.csv', delimiter=',', skiprows=1
).T
T_data_K = T_data_C + 273.15
D_th_data_SI = D_th_data_scaled * 1e-12

# Sort the table before interpolation.
order = np.argsort(T_data_K)
T_data_K = T_data_K[order]
D_th_data_SI = D_th_data_SI[order]

print(D_th_data_SI)
print(T_data_K)

# ---------------------- Temperature profile fit ----------------------
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
plt.xlabel('d (µm)')
plt.ylabel('T (K)')
plt.legend()
plt.show()

# ---------------------- Physical parameters ----------------------
np.random.seed(24)

R = 2.05e-6
eta = 1.0e-3
T_ref = 300.0
k_B = 1.3806e-23

gamma = 6 * np.pi * eta * R
D_t = k_B * T_ref / gamma

# Fixed temperature-fit constants retained from the original simulation.
A_fit = 47487805.196155205
k_fit = 4.262380207691129
C_fit = 297.5165761061297

# ---------------------- Pulse schedule ----------------------
period = 100e-6
pulse_start = 0.0
pulse_width = 10e-9
tau_cool = 100e-6

# ---------------------- Time steps ----------------------
dt_on = 1e-9
dt_off = 1e-6

total_time = 5.0
num_periods = int(np.round(total_time / period))

n_on = int(np.round(pulse_width / dt_on))

# The final OFF step is shorter, so every cycle lasts exactly one period.
off_duration = period - n_on * dt_on
n_off_full = int(np.floor(off_duration / dt_off + 1e-10))
off_remainder = off_duration - n_off_full * dt_off

if off_remainder < 1e-15:
    off_remainder = 0.0
if off_remainder < 0:
    raise ValueError('Invalid OFF remainder')

save_every_periods = 10

# ---------------------- Initial position ----------------------
r0_um = 3.35
r = np.array([r0_um, 0.0]) * 1e-6
positions = [r.copy()]

# ---------------------- Functions ----------------------
def temperature_and_grad(r_xy, A_now):
    x, y = r_xy
    rr = np.hypot(x, y)
    rr_um = rr * 1e6

    T_local = A_now * np.exp(-k_fit * rr_um) + C_fit

    # Convert temperature gradient from K/µm to K/m.
    dT_dr = -A_now * k_fit * np.exp(-k_fit * rr_um) * 1e6

    if rr == 0.0:
        grad_T_local = np.zeros(2)
    else:
        grad_T_local = dT_dr * r_xy / rr

    return T_local, grad_T_local


def D_th(T_local):
    # Use endpoint values outside the tabulated temperature range.
    return np.interp(T_local, T_data_K, D_th_data_SI)


def decay_amp(A_now, dt):
    return A_now * np.exp(-dt / tau_cool)


# ---------------------- Simulation ----------------------
FILE_PREFIX = 'pulsed'
A_t = 0.0

for p in range(num_periods):

    # Laser ON.
    A_t = A_fit

    for i in range(n_on):
        T_local, grad_T_local = temperature_and_grad(r, A_t)

        drift = -D_th(T_local) * grad_T_local * dt_on
        brown = np.sqrt(2 * D_t * dt_on) * np.random.normal(size=2)

        r = r + drift + brown

    # Laser OFF: exponentially decaying cooling.
    off_steps = [dt_off] * n_off_full

    if off_remainder:
        off_steps.append(off_remainder)

    for dt in off_steps:
        A_t = decay_amp(A_t, dt)
        T_local, grad_T_local = temperature_and_grad(r, A_t)

        drift = -D_th(T_local) * grad_T_local * dt
        brown = np.sqrt(2 * D_t * dt) * np.random.normal(size=2)

        r = r + drift + brown

    if p % save_every_periods == 0:
        positions.append(r.copy())


# ---------------------- MSD calculation ----------------------
positions = np.array(positions)
positions_um = positions * 1e6

# The initial position is t=0; subsequent positions are saved after
# periods 1, 11, 21, ... because p is zero-indexed.
saved_periods = np.r_[0, np.arange(1, num_periods + 1, save_every_periods)]
times = saved_periods * period

assert len(times) == len(positions_um)

# Khoshzaban et al., Eq. (1):
# MSD_N = (1/N) * sum_{i=1}^N |r_i - r_0|^2

# Squared displacement relative to the initial position.
dr0_um = positions_um[1:] - positions_um[0]
squared_displacement = np.sum(dr0_um**2, axis=1)

# Cumulative mean squared displacement.
N = np.arange(1, len(squared_displacement) + 1)
msd_khoshzaban = np.cumsum(squared_displacement) / N
msd_times = times[1:]

# ---------------------- Diagnostics ----------------------
print('Total simulated time (s):', num_periods * period)
print('Periods:', num_periods)
print('Steps per period:', n_on + n_off_full + int(off_remainder > 0))
print('Approx total steps:', num_periods * (
    n_on + n_off_full + int(off_remainder > 0)
))
print('Saved points:', len(positions_um))
print('Final Khoshzaban MSD (µm²):', msd_khoshzaban[-1])

# ---------------------- MSD plotting ----------------------
plt.figure(figsize=(10, 6))

plt.plot(
    msd_times,
    msd_khoshzaban,
    color='red',
    lw=1.1,
    label='Khoshzaban MSD'
)

plt.xlabel('Time (s)')
plt.ylabel(r'MSD ($\mu\mathrm{m}^2$)')
plt.title('MSD for pulsed laser simulation')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(FILE_PREFIX + '-msd.pdf')
plt.show()


# ---------------------- Plotting ----------------------
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
    label='Trajectory',
)

plt.scatter(
    positions_um[0, 0],
    positions_um[0, 1],
    s=70,
    c='green',
    label='Start',
    zorder=10,
)

plt.scatter(
    positions_um[-1, 0],
    positions_um[-1, 1],
    s=70,
    c='purple',
    label='End',
    zorder=10,
)

# Laser boundaries (visualization only).
laser_waist_um = 1.35

for laser_radius_um in (laser_waist_um, 2 * laser_waist_um):
    plt.gca().add_patch(
        plt.Circle(
            (0.0, 0.0),
            laser_radius_um,
            fill=False,
            color='orange',
            linestyle='--',
            linewidth=1.5,
            label=f'Laser radius {laser_radius_um:.2f} µm',
            zorder=4,
        )
    )

# Initial particle boundary (visualization only).
plt.gca().add_patch(
    plt.Circle(
        (r0_um, 0.0),
        R * 1e6,
        fill=False,
        color='green',
        linestyle='--',
        linewidth=1.5,
        label='Initial particle boundary',
        zorder=4,
    )
)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

plt.savefig(FILE_PREFIX + '.pdf')
plt.show()
