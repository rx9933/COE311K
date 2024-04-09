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
        raise ValueError("h must be positive")
    
    dfxi = np.zeros_like(xi)
    
    for i in range(len(xi)):
        if i < 2 or i > len(xi) - 3:
            # Use forward/backward difference for points near the boundaries
            if i < 2:
                dfxi[i] = (-f(xi[i+2]) + 8*f(xi[i+1]) - 8*f(xi[i-1]) + f(xi[i-2])) / (12*h)
            else:
                dfxi[i] = (-f(xi[i+2]) + 8*f(xi[i+1]) - 8*f(xi[i-1]) + f(xi[i-2])) / (12*h)
        else:
            dfxi[i] = (-f(xi[i+2]) + 8*f(xi[i+1]) - 8*f(xi[i-1]) + f(xi[i-2])) / (12*h)
    
    return dfxi
