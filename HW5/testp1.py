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
