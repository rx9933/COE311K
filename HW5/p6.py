import numpy as np

# Function to integrate
def f(x):
    return 2*x*9-x**4

# Trapezoidal rule
def trapezoidal_rule(a, b, f, n_intervals):
    h = (b - a) / n_intervals
    x = np.linspace(a, b, n_intervals + 1)
    result = 0.5 * h * (f(a) + f(b))
    result += h * np.sum(f(x[1:-1]))
    return result, n_intervals

# Simpson's 1/3 rule
def simpsons_13_rule(a, b, f, n_intervals):
    if n_intervals % 2 != 0:
        raise ValueError("Number of intervals must be even for Simpson's rule")
    h = (b - a) / n_intervals
    x = np.linspace(a, b, n_intervals + 1)
    result = f(a) + f(b)
    result += 4 * np.sum(f(x[1:-1:2]))
    result += 2 * np.sum(f(x[2:-2:2]))
    result = 1/3 * h * result
    return result, n_intervals

# Interval
a = 0
b = 1
n_intervals = 2  # Number of intervals

# Trapezoidal rule result
trapezoidal_result, trapezoidal_intervals = trapezoidal_rule(a, b, f, n_intervals)

# Simpson's 1/3 rule result
simpsons_result, simpsons_intervals = simpsons_13_rule(a, b, f, n_intervals)

print("Trapezoidal rule result:", trapezoidal_result, "using", trapezoidal_intervals, "intervals")
print("Simpson's 1/3 rule result:", simpsons_result, "using", simpsons_intervals, "intervals")
print("Actual value of x**2 for 0 to 1 is ", 8.80)