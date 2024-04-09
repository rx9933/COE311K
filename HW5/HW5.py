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

