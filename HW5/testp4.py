def f(x):
    return np.exp(-x)
from matplotlib import pyplot as plt
# Given values
xi = np.array([0.6])
h_values = np.array([1,.5,.25,.125])
exact_derivative = -np.exp(-0.6)

# Compute errors
errors = []
for h in h_values:
    numerical_derivative = fourth_order_diff(f, xi, h)[0]
    errors.append(np.abs(numerical_derivative - exact_derivative))

# Compute convergence rates
rates = []
for i in range(1, len(errors)):
    rate = np.log(errors[i] / errors[i-1]) / np.log(h_values[i] / h_values[i-1])
    rates.append(rate)

# Plot h vs error
plt.figure()
plt.plot(h_values, errors, marker='o')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('h')
plt.ylabel('Error')
plt.title('Convergence Analysis')
plt.grid()
plt.show()
plt.savefig("p4")
# Print convergence rates
print("Convergence rates:", rates)
for o in range(len(errors)-1):
    print(errors[o]/errors[o+1])
