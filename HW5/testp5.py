
# Testing the function with a constant function
x_const = np.array([0, 1, 2, 3, 4])
fx_const = np.array([5, 5, 5, 5, 5])
I_const = my_composite_trap(x_const, fx_const)
print("Integral of constant function (should be 20):", I_const)

# Testing the function with a linear function
x_linear = np.array([0, 1, 2, 3, 4])
fx_linear = np.array([0, 1, 2, 3, 4])
I_linear = my_composite_trap(x_linear, fx_linear)
print("Integral of linear function (should be 8):", I_linear)

# Testing the function with a quad function
x_quad=np.linspace(0,4,100)
fx_quad = x_quad**2
I_quad = my_composite_trap(x_quad, fx_quad)
print("Integral of linear function (should be 21.33):", I_quad)