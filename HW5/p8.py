import numpy as np
import matplotlib.pyplot as plt

def solve_BVP_FD(T0, T1, k, dx):
    x = np.arange(0, 1 + dx, dx)
    N = len(x)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Construct the coefficient matrix A and the right-hand side vector b
    for i in range(1, N - 1):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1
        b[i] = -dx**2 * x[i] / k
    
    # Apply the boundary conditions
    A[0, 0] = 1
    b[0] = T0
    A[N - 1, N - 1] = 1
    b[N - 1] = T1  
    print(A)
    # Solve the linear system
    T = np.linalg.solve(A, b)
    print(b)
    print(T)
    return x, T

# Analytical solution
def analytical_solution(x, k):
    return -(x**3) / (6*k) + (1 + 1/(6*k)) * x

# Example usage
T0 = 0
T1 = 1
k = 0.1
dx = 0.5
x, T = solve_BVP_FD(T0, T1, k, dx)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, T, label='Finite Difference Solution')
plt.plot(x, analytical_solution(x, k), label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Temperature Distribution')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("p8")