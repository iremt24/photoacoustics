import numpy as np
import matplotlib.pyplot as plt

T_data_C, D_th_data_scaled = np.loadtxt('braibanti-fig3.csv', delimiter=',', skiprows=1).T
T_data_K = T_data_C + 273.15
D_th_data_SI = D_th_data_scaled * 1e-12

# Interpolation uses Kelvin; mobility is in m^2/(s*K).
def D_th(T_local):
    """Thermophoretic mobility [m^2/(s*K)] at T_local [K]; clamp to data range."""
    return np.interp(T_local, T_data_K, D_th_data_SI)

# Make smooth temperature range, to plot visually
T_fine_C = np.linspace(T_data_C.min(), T_data_C.max(), 200)

plt.scatter(T_data_C, D_th_data_SI * 1e12, label='Digitized data')
plt.plot(T_fine_C, D_th(T_fine_C + 273.15) * 1e12, label='Interpolated thermal mobility')

plt.xlabel('Temperature (°C)')
plt.ylabel(r'$D_{\mathrm{th}}\ (10^{-12}\,\mathrm{m^2\,s^{-1}\,K^{-1}})$')
plt.legend()
plt.grid(True)
plt.savefig('d_th.pdf')
plt.show()
