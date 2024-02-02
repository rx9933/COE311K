import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, tanh, factorial, pi

# problem 1
def exact_velocity(cd, m, t, g):
    v = np.zeros(len(t))
    for timeindex, timevalue in enumerate(t):
        v[timeindex] = sqrt(g*m/cd)*tanh(sqrt(g*cd/m)*timevalue)
    return np.array(v)

# problem 3
def forward_Euler_velocity(cd, m, t, g):
    numSteps = len(t)
    v = np.zeros(numSteps)
    stepNo = 1
    while stepNo < numSteps:
        dt = t[stepNo]-t[stepNo-1]
        v[stepNo] = v[stepNo-1] + dt*(g-cd/m*(v[stepNo-1])**2)
        stepNo+=1
    return np.array(v)

# problem 5
def root_mean_square_error(maxt, dt, cd, m, t, g):
    tsteps = int(maxt/dt) + 1
    t = np.linspace(0,maxt, tsteps)
    dv = np.array(exact_velocity(cd, m, t, g) - forward_Euler_velocity(cd, m, t, g))
    return sqrt(np.sum(dv**2)/tsteps)

# bonus, part1
def mat_mat_mul(matrix1, matrix2):
    # Initialize result matrix with zeros
    result = np.zeros((matrix1.shape[0], matrix2.shape[1])) # shape should be rows of first matrix and columns of second matrix
    # Perform matrix multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return np.array(result)

def approximate_sin(x, es, maxit):

    # Initialize variables
    approximation = 0
    old_approximation = 0
    relerror = 100
    iter = 0
    # Loop until the maximum number of iterations or until the error is less than the allowed error
    while relerror > es and iter < maxit:
        old_approximation = approximation
        approximation += ((-1)**iter) * (x**(2*iter+1)) / factorial(2*iter+1)
        if abs(approximation) > 10**-7: # prevent any division by 0.0 (or close value)
            relerror = abs((approximation - old_approximation) / approximation)
        iter += 1

    return approximation

# problem 1 
t = np.linspace(0,12,25)
cd = .25
m = 68.1
g = 9.81
v_exact = exact_velocity(cd,m,t,g)
# problem 3
v_euler = forward_Euler_velocity(cd,m,t,g)

# problem 2
plt.figure(0)
plt.plot(t, v_exact,
           color = "darkcyan",
           markerfacecolor = "lightskyblue",
           marker = "o", markersize = 3.5,
           label = "Exact Velocity")
plt.title("Velocity vs. Time")
plt.xlabel("Time (s)") # assumed seconds
plt.ylabel("Velocity (m/s)") # assumed meters/second
plt.legend([f"Exact Velocity of Object\n
            (mass = {m} kg and 
            drag coefficient = {cd})"])
plt.grid()
plt.show()
plt.savefig("HW1P2.png")


# problem 4 
plt.figure(1)
plt.plot(t, v_exact,  color = "darkcyan", markerfacecolor = "lightskyblue",marker = "o", markersize = 3.5, label = "Exact Velocity")
plt.plot(t, v_euler,  color = "red", markerfacecolor = "lightcoral", marker = "o", markersize = 3.5, label = "Euler-Calculuated Velocity")
plt.title("Velocity vs. Time")
plt.xlabel("Time (s)") # assumed to be in seconds
plt.ylabel("Velocity (m/s)") # assumed to be in meters per second
plt.legend([f"Exact Velocity of Object\n(mass = {m} kg and drag coefficient = {cd})", f"Euler-Calculated Velocity of Object\n(mass = {m} kg and drag coefficient = {cd})"])
plt.grid()
plt.show()
plt.savefig("HW1P4.png")


# problem 5
rmse = []
step_sizes = [.0625,.125,.25,.5,1,2]
max_time = 12
for dt in step_sizes:
    rmse+=[root_mean_square_error(max_time,dt,cd,m,t,g)]
plt.figure(2)
plt.plot(step_sizes, rmse,
           color = "darkgreen", 
           markerfacecolor = "darkorange",
           marker = "o", markersize = 5, 
           label = "Exact Velocity")
plt.title("Root Mean Square Error (RMSE) vs. Time Step Size")
plt.xlabel("Time (s)") # assumed to be in seconds
plt.ylabel("RMSE (m/s)") # assumed to be in meters per second
plt.legend([f"Root Mean Square Error for Euler-Calculated Velocity
            \n(mass = {m} kg and drag coefficient = {cd})"])
plt.grid()
plt.show()
plt.savefig("HW1P5.png")


# bonus, part1
rng = np.random.default_rng(12345)
rints = rng.integers(low=0, high=10, size=2)

matrix1_rows, matrix2_rows = (rints) # rows of matrix 1 are columns in matrix 2; rows of matrix 2 are columns in matrix 1
matrix1 = np.random.rand(matrix1_rows,matrix2_rows)
matrix2 = np.random.rand(matrix2_rows,matrix1_rows)

explicit = mat_mat_mul(matrix1,matrix2)
actual = matrix1@matrix2
print("mat_mat_mul", explicit)
print("actual", actual)
print("No difference between matrixes: ", np.allclose(explicit, actual))

# bonus, part2
print("approximate sin value:", approximate_sin(pi, .0001, 5))