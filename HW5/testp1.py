def true_function(x):
    # return x**4-x**3+x**2  # Example true function (f(x) = x^2)
    return -np.sin(x)

# Generate some example data
x = np.linspace(-10, 10, 10)
fx = true_function(x)
# Calculate finite differences
fd_types = ['Forward', 'Backward', 'Centered']
colors = ['r', 'g', 'b']
plt.figure(figsize=(10, 6))
# fx = x**4-x**3+x**2

plt.plot(x, -np.cos(x), label='True Function', color='black')
for fd_type, color in zip(fd_types, colors):
    dfxi = my_finite_diff(fx, x, fd_type)
    plt.plot(x[1:-1], dfxi, label=f'{fd_type} Finite Difference', color=color)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Finite Difference Approximations vs. True Function')
plt.legend()
plt.grid(True)
plt.show()
##########
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