import numpy as np
error = "error"

# Problem 2
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

# Problem 3
def fourth_order_diff(f, xi, h):
    if h <= 0:
        return error
    dfxi = np.zeros_like(xi)
    for i in range(len(xi)):
        dfxi[i] = (-f(xi[i]+2*h) + 8*f(xi[i]+1*h) - 8*f(xi[i]-h) + f(xi[i]-2*h)) / (12*h)
    return dfxi

# Problem 5
def my_composite_trap(x, fx):
    if len(x) != len(fx):
        raise ValueError("x and fx must have the same length")
    n = len(x) - 1
    h = np.diff(x)
    A = 0
    for i in range(len(x)-1):
        A+=h[i]*(fx[i]+fx[i+1])/2
    return A

