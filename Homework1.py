import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, tanh

def exact_velocity(cd, m, t, g):
    v = np.zeros(len(t))
    for timeindex, timevalue in enumerate(t):
        v[timeindex] = sqrt(g*m/cd)*tanh(sqrt(g*cd/m)*timevalue)
    return np.array(v)

def forward_Euler_velocity(cd, m, t, g):
    numSteps = len(t)
    v = np.zeros(numSteps)
    stepNo = 1
    while stepNo < numSteps:
        dt = t[stepNo]-t[stepNo-1]
        v[stepNo] = v[stepNo-1] + dt*(g-cd/m*(v[stepNo-1])**2)
        stepNo+=1
    return np.array(v)

t = np.linspace(0,12,25)
cd = .25
m = 68.1
g = 9.81
v_exact = exact_velocity(cd,m,t,g)
v_euler = forward_Euler_velocity(cd,m,t,g)

plt.plot(t, v_exact,  color = "darkcyan", markerfacecolor = "lightskyblue",marker = "o", markersize = 3.5, label = "Exact Velocity")
plt.plot(t, v_euler,  color = "red", markerfacecolor = "lightcoral", marker = "o", markersize = 3.5, label = "Euler-Calculuated Velocity")
plt.title("Velocity vs. Time")
plt.xlabel("Time (s)") # assumed to be in seconds
plt.ylabel("Velocity (m/s)") # assumed to be in meters per second
plt.legend([f"Exact Velocity of Object\n(mass = {m} kg and drag coefficient = {cd})", f"Euler-Calculated Velocity of Object\n(mass = {m} kg and drag coefficient = {cd})"])
plt.grid()
plt.show()
plt.savefig("HW1P4.png")
