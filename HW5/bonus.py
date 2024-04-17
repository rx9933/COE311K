import numpy as np
from matplotlib import pyplot as plt

def solve_freefall_RK4(x0, v0, nt, dt, g, cd, m):
    def rhs_func_x(t, v):
        return v

    def rhs_func_v(t, v):
        return g - cd/m * v**2

    x_RK4 = np.zeros(nt+1)
    v_RK4 = np.zeros(nt+1)

    x_RK4[0] = x0
    v_RK4[0] = v0

    for i in range(nt):
        t = i * dt

        k1_x = rhs_func_x(t, v_RK4[i])
        k1_v = rhs_func_v(t, v_RK4[i])

        k2_x = rhs_func_x(t + dt/2, v_RK4[i] + k1_v * dt/2)
        k2_v = rhs_func_v(t + dt/2, v_RK4[i] + k1_v * dt/2)

        k3_x = rhs_func_x(t + dt/2, v_RK4[i] + k2_v * dt/2)
        k3_v = rhs_func_v(t + dt/2, v_RK4[i] + k2_v * dt/2)

        k4_x = rhs_func_x(t + dt, v_RK4[i] + k3_v * dt)
        k4_v = rhs_func_v(t + dt, v_RK4[i] + k3_v * dt)

        x_RK4[i+1] = x_RK4[i] + dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_RK4[i+1] = v_RK4[i] + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return x_RK4, v_RK4

# Given parameters
x0 = 0
v0 = 0
nt = 10
g = 9.81
cd = 0.5
m = 1
t=10
# Time step sizes to analyze
dt_values = [2, 1, 0.5, 0.25, 0.125, 0.0625]

x_true = m/cd*np.log(np.cosh(np.sqrt(g*cd/m)*t))
v_true = np.sqrt(g*m/cd) * np.tanh((np.sqrt(g*cd/m)*t))


# Compute errors and convergence rates
x_errors = []
v_errors = []
rates = []
for dt in dt_values:
    nt = int(10 / dt)
    x_RK4, v_RK4 = solve_freefall_RK4(x0, v0, nt, dt, g, cd, m)
    x_errors.append(np.abs(x_RK4-x_true))
    v_errors.append(np.abs(v_RK4-v_true))
    
xrates = []
vrates = []
for i in range(1,len(x_errors)):
    x_rates = np.log(x_errors[i]/x_errors[i-1]) / np.log(h_values[i]/h_values[i-1])
    v_rates = np.log(v_errors[i]/v_errors[i-1]) / np.log(h_values[i]/h_values[i-1])


# Plot errors for position and velocity
xerrors = np.array(xerrors)
verrors = np.array(verrors)

plt.figure()
plt.plot(dt_values, xerrors, marker='o', label='Position error')
plt.plot(dt_values, verrors, marker='o', label='Velocity error')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time Step Size (dt)')
plt.ylabel('Error')
plt.title('Convergence Analysis for Free Fall Problem')
plt.legend()
plt.grid()
plt.show()
"""
# Compute convergence rates
# Compute convergence rates
xrates = []
for i in range(1, len(dt_values)):
    rate = np.log(np.max(errors[i] / errors[i-1])) / np.log(dt_values[i-1] / dt_values[i])
    rates.append(rate)


# Print convergence rates
print("position convergence rates:", xrates)

"""
print("xrate", x_rates)
print("vrate",v_rates)