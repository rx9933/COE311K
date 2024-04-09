import numpy as np
error = "error"

def my_finite_diff(fx, xi, fd_type):
    if len(fx) != len(xi):
        raise ValueError("Lengths of fx and xi should be the same")
    n = len(fx)
    dfxi = np.zeros(n-2)
    if fd_type == 'Forward':
        for i in range(0, n-2):
            dfxi[i] = (fx[i+1] - fx[i]) / (xi[i+1] - xi[i])
    elif fd_type == 'Backward':
        for i in range(1, n-1):
            dfxi[i-1] = (fx[i] - fx[i-1]) / (xi[i] - xi[i-1])
    elif fd_type == 'Centered':
        for i in range(1, n-1):
            dfxi[i-1] = (fx[i+1] - fx[i-1]) / (xi[i+1] - xi[i-1])
    else:
        raise ValueError("Invalid fd_type. Use 'Forward', 'Backward', or 'Centered'")
    
    return dfxi

from matplotlib import pyplot as plt
# step size
h = 0.1
# define grid
x = np.arange(0, 2*np.pi, h) 
# compute function
y = np.cos(x) 

# compute vector of forward differences
# forward_diff = np.diff(y)/h 
forward_diff = my_finite_diff(y, x, "Forward")
# compute corresponding grid
x_diff = x[:-1:] 
# compute exact solution
exact_solution = -np.sin(x_diff) 

# Plot solution
plt.figure(figsize = (12, 8))
plt.plot(x[1:-1], forward_diff, '--', label = 'Finite difference approximation')
plt.plot(x_diff, exact_solution, \
         label = 'Exact solution')
plt.legend()
plt.show()

# Compute max error between 
# numerical derivative and exact solution
max_error = max(abs(exact_solution - forward_diff))
print(max_error)