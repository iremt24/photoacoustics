# ---------------------- Imports ----------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from matplotlib.colors import LogNorm

# ---------------------- Load and convert mobility data ----------------------
# This CSV has no header.
T_data_C, D_th_data_scaled = np.loadtxt("braibanti-fig3.csv", delimiter=",").T
T_data_K = T_data_C + 273.15
D_th_data_SI = D_th_data_scaled * 1e-12

# ---------------------- Diagnostics ----------------------
print(D_th_data_SI)
print(T_data_K)

# ---------------------- Temperature simulation data ----------------------
d = np.array([3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 5.0])
T_C = np.array([48, 41, 35, 31, 28, 27, 26, 25])

# Convert Celsius to Kelvin
T = T_C + 273.15

# ---------------------- Spatial temperature model ----------------------
# T(x) = A * exp(-k x) + C
def model(x, A, k, C):
    return A * np.exp(-k * x) + C

# ---------------------- Initial guesses ----------------------
A0 = T[0] - T[-1]   # amplitude (K)
k0 = 1.0            # decay rate guess
C0 = T[-1]          # baseline (K)
p0 = [A0, k0, C0]

# ---------------------- Fit ----------------------
popt, pcov = curve_fit(model, d, T, p0=p0)
A, k, C = popt

# ---------------------- Diagnostics ----------------------
print("Fit parameters:")
print("A =", A)
print("k =", k)
print("C =", C)

# ---------------------- Plotting ----------------------
xfit = np.linspace(min(d), max(d), 200)
yfit = model(xfit, *popt)

plt.scatter(d, T, label="data (K)")
plt.plot(xfit, yfit, label="exp fit (K)")
plt.xlabel("d")
plt.ylabel("T (K)")
plt.legend()
plt.show()

# ---------------------- Load COMSOL data ----------------------
# COMSOL temperature histories: exported times are in ns, temperatures in K.
def load_temperature(filename):
    with open(filename) as file:
        for line in file:
            if "@ t=" in line:
                time = np.array(re.findall(r"@ t=([^\s]+)", line), dtype=float) * 1e-9
                break
        else:
            raise ValueError(f"No COMSOL time header found in {filename}")
    temperature = np.loadtxt(filename, comments="%").flatten()
    if time.size != temperature.size:
        raise ValueError(f"Time and temperature lengths differ in {filename}")
    return time, temperature

# ---------------------- Plotting COMSOL data ----------------------
fig_comsol, axes_comsol = plt.subplots(1, 2, figsize=(12, 4))
comsol_profiles = [
    ("3.4 µm", "Gold-50nJ-MAX.txt", "Gold-50nJ-Min.txt"),
    ("5 µm", "Gold-50nJ-MAX-5um.txt", "Gold-50nJ-MIN-5um.txt"),
]
for ax_comsol, (distance_label, max_file, min_file) in zip(axes_comsol, comsol_profiles):
    for filename, label in [(max_file, "Maximum"), (min_file, "Minimum")]:
        profile_time, profile_temperature = load_temperature(filename)
        ax_comsol.plot(profile_time * 1e6, profile_temperature, label=label)
    ax_comsol.set(title=f"Distance: {distance_label}", xlabel="Time (µs)")
    ax_comsol.grid(True, alpha=0.3)
    ax_comsol.legend()
for ax_comsol in axes_comsol:
    ax_comsol.set_ylabel("Temperature (K)")
fig_comsol.suptitle("COMSOL temperature response")
fig_comsol.tight_layout()
plt.savefig("comsol-response.pdf")
plt.show()

# ---------------------- Random seed ----------------------
np.random.seed(24)

# ---------------------- Parameters ----------------------
R = 2.05e-6
eta = 1.0e-3
T_ref = 300.0
k_B = 1.3806e-23
gamma = 6 * np.pi * eta * R
D_t = k_B * T_ref / gamma

def D_th(temperature):
    # Linear interpolation; use endpoint mobility outside the CSV range.
    return np.interp(temperature, T_data_K, D_th_data_SI)

period = 100e-6
total_time = 100
num_periods = int(round(total_time / period))
save_every_periods = 20  # Output sampling only; does not change the dynamics.

# ---------------------- COMSOL data ----------------------
# Temperatures are in K; exported times are in ns.
time, T_max = load_temperature("Gold-50nJ-MAX.txt")
time_min, T_min = load_temperature("Gold-50nJ-Min.txt")
time_5, T_max_5 = load_temperature("Gold-50nJ-MAX-5um.txt")
time_min_5, T_min_5 = load_temperature("Gold-50nJ-MIN-5um.txt")
if not (np.array_equal(time, time_min) and np.array_equal(time, time_5)
        and np.array_equal(time, time_min_5)):
    raise ValueError("COMSOL files must have matching time grids.")

delta_T = T_max - T_min
delta_T_5 = T_max_5 - T_min_5

# ---------------------- Jiang propulsion ----------------------
# Mean of the near/far temperatures is a proxy for particle temperature.
T_particle = 0.5 * (T_max + T_min)
T_particle_5 = 0.5 * (T_max_5 + T_min_5)
v_ref = D_th(T_particle) * delta_T / (3 * R)
v_5 = D_th(T_particle_5) * delta_T_5 / (3 * R)
displacement_ref = np.trapezoid(v_ref, time)
displacement_5 = np.trapezoid(v_5, time)

# Heating/cooling is already in COMSOL; assume zero propulsion after 10 us.

# Average is printed as a diagnostic, the simulation uses the resolved velocity response.
v_mean_ref = displacement_ref / period

# Temperature rise decays with distance.
d_ref_um = 3.4
d_far_um = 5.0

# ---------------------- Time steps within each pulse ----------------------
# Resolve rapid heating and early cooling, then use coarser late-cooling steps.
dt_fast = 1e-9       # 1 ns, from 0 to 200 ns
dt_cool = 10e-9      # 10 ns, from 200 ns to 1 us
dt_tail = 100e-9     # 0.1 us, from 1 us to the COMSOL endpoint (10 us)
# Boundaries for the three parts of the COMSOL response (seconds).
t_fast = np.arange(200) * dt_fast
t_cool = 200e-9 + np.arange(80) * dt_cool
t_tail = 1e-6 + np.arange(90) * dt_tail
pulse_edges = np.concatenate([t_fast, t_cool, t_tail, [time[-1]]]) # contains times
pulse_dt = np.diff(pulse_edges)                                    # contains durations
if time[0] != 0 or not np.all(pulse_dt > 0) or time[-1] >= period:
    raise ValueError("Pulse grid must fit inside the period and start at zero.")

# Sample the COMSOL temperatures at the midpoint of each simulation interval.
# Fit separate exponential temperature-rise profiles at every midpoint so that
# both the reference and 5 um histories are reproduced during heating/cooling.
pulse_midpoints = 0.5 * (pulse_edges[:-1] + pulse_edges[1:])
rise_max_ref = np.maximum(np.interp(pulse_midpoints, time, T_max) - T_ref, 0.0)
rise_min_ref = np.maximum(np.interp(pulse_midpoints, time, T_min) - T_ref, 0.0)
rise_max_5 = np.maximum(np.interp(pulse_midpoints, time, T_max_5) - T_ref, 0.0)
rise_min_5 = np.maximum(np.interp(pulse_midpoints, time, T_min_5) - T_ref, 0.0)

# A tiny floor avoids taking log(0) for temperatures at ambient within precision.
rise_floor = 1e-9  # K
k_max = np.log(np.maximum(rise_max_ref, rise_floor)
               / np.maximum(rise_max_5, rise_floor)) / (d_far_um - d_ref_um)
k_min = np.log(np.maximum(rise_min_ref, rise_floor)
               / np.maximum(rise_min_5, rise_floor)) / (d_far_um - d_ref_um)

def local_temperatures(distance_um, i):
    offset_um = distance_um - d_ref_um
    local_max = T_ref + rise_max_ref[i] * np.exp(-k_max[i] * offset_um)
    local_min = T_ref + rise_min_ref[i] * np.exp(-k_min[i] * offset_um)
    return local_max, local_min

# ---------------------- Brownian diffusion ----------------------
# Brownian prefactors for the response and the remaining OFF time.
sqrt_2D_pulse = np.sqrt(2 * D_t * pulse_dt)
cold_time = period - time[-1]
sqrt_2D_cold = np.sqrt(2 * D_t * cold_time)

# ---------------------- Initial position ----------------------
laser_waist_um = 1.35
r0_um = laser_waist_um + R * 1e6
r = np.array([r0_um, 0.0]) * 1e-6
theta = 0.0  # fixed orientation in radians; no rotational diffusion
positions = [r.copy()]

# Record the initial state, then every selected number of pulse periods.
if not isinstance(save_every_periods, (int, np.integer)) or save_every_periods < 1:
    raise ValueError("save_every_periods must be a positive integer.")
saved_times_constant_D_90s = [0.0]

# ---------------------- Simulation ----------------------
# Fixed propulsion direction, as in the original fixed-angle model.
direction = np.array([np.cos(theta), np.sin(theta)])
n_response = len(pulse_dt)

for p in range(num_periods):

    # --- Heating and cooling: use the full COMSOL response ---
    noise = np.random.normal(size=(n_response, 2))
    for i in range(n_response):
        dt = pulse_dt[i]
        distance_um = np.linalg.norm(r) * 1e6
        local_max, local_min = local_temperatures(distance_um, i)
        particle_temperature = 0.5 * (local_max + local_min)
        speed = D_th(particle_temperature) * (local_max - local_min) / (3 * R)

        thermophoretic_displacement = speed * direction * dt
        random_displacement = sqrt_2D_pulse[i] * noise[i]
        r = r + thermophoretic_displacement + random_displacement

    # --- Remaining OFF time: Brownian motion with zero propulsion ---
    random_displacement = sqrt_2D_cold * np.random.normal(size=2)
    r = r + random_displacement

    # Always record the final state, even if the sampling interval does not divide the run.
    if (p + 1) % save_every_periods == 0 or p + 1 == num_periods:
        positions.append(r.copy())
        saved_times_constant_D_90s.append((p + 1) * period)

# ---------------------- Results ----------------------
positions_constant_D_90s_um = np.array(positions) * 1e6
saved_times_constant_D_90s = np.array(saved_times_constant_D_90s)

# ---------------------- Diagnostics ----------------------
print("Simulated time (s):", total_time)
print("Brownian diffusion (um^2/s):", D_t * 1e12)
print("Temperature decay rates vary with pulse time (1/um).")
print("Reference mean Jiang speed (um/s):", v_mean_ref * 1e6)
print("Resolved heating/cooling steps per pulse:", len(pulse_dt))
print("Total position updates:", num_periods * (len(pulse_dt) + 1))

# ---------------------- Plotting ----------------------
fig, ax = plt.subplots(figsize=(12, 8))
xmin = min(-3.0, positions_constant_D_90s_um[:, 0].min() - 0.5)
xmax = max(8.0, positions_constant_D_90s_um[:, 0].max() + 0.5)
ymin = min(-3.0, positions_constant_D_90s_um[:, 1].min() - 0.5)
ymax = max(3.0, positions_constant_D_90s_um[:, 1].max() + 0.5)

# The spatial temperature fit provides the background, only for visualization.
A_fit, k_fit, C_fit = popt
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 320), np.linspace(ymin, ymax, 320))
TT = A_fit * np.exp(-k_fit * np.sqrt(xx**2 + yy**2)) + C_fit
im = ax.imshow(TT, extent=[xmin, xmax, ymin, ymax], origin="lower",
               cmap="coolwarm", norm=LogNorm(vmin=TT.min(), vmax=TT.max()), alpha=0.8)
fig.colorbar(im, ax=ax, label="Original spatial fit (K, log scale)", shrink=0.6)

for radius in [laser_waist_um, 2 * laser_waist_um]:
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="orange", ls="--",
                           label=f"Laser radius {radius:.2f} um"))
ax.add_patch(plt.Circle((r0_um, 0), R * 1e6, fill=False, color="green", ls="--",
                       label="Initial Janus surface"))
ax.plot(positions_constant_D_90s_um[:, 0], positions_constant_D_90s_um[:, 1], color="black", lw=2.5)
ax.scatter(*positions_constant_D_90s_um[0], color="green", s=100, label="start", zorder=5)
ax.scatter(*positions_constant_D_90s_um[-1], color="purple", s=100, label="end", zorder=5)
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (µm)", ylabel="y (µm)",
       title="Brownian motion + COMSOL-based Jiang propulsion")
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig("pulsed-laser.png")
plt.show()


# ---------------------- MSD calculation ----------------------
# Khoshzaban et al., Eq. (1): cumulative mean squared displacement
# relative to the initial position, using the saved trajectory samples.
# The initial sample is the reference position and is not counted in N.
displacements_from_initial_um = (
    positions_constant_D_90s_um[1:] - positions_constant_D_90s_um[0]
)
squared_displacements_um2 = np.sum(displacements_from_initial_um**2, axis=1)
msd_um2 = np.cumsum(squared_displacements_um2) / np.arange(
    1, len(squared_displacements_um2) + 1
)
msd_times_s = saved_times_constant_D_90s[1:]

# ---------------------- MSD diagnostics ----------------------
print("Final MSD (um^2):", msd_um2[-1])
print("Final squared displacement from start (um^2):", squared_displacements_um2[-1])

# ---------------------- MSD plotting ----------------------
fig_msd, ax_msd = plt.subplots(figsize=(10, 6))
ax_msd.plot(msd_times_s, msd_um2, color="red", lw=2)
ax_msd.set(
    xlabel="Time (s)",
    ylabel="MSD (µm²)",
)
ax_msd.grid(True, linestyle="--", alpha=0.4)
fig_msd.tight_layout()
fig_msd.savefig("self-thermophoresis-msd.pdf")
plt.show()
