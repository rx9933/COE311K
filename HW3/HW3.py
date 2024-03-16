import numpy as np
error = "error"

# Problem 2
def least_squares_poly(x,y,k):
    m = len(x)
    n = k
    if (len(x) != len(y)):
        return error

    Z = np.zeros((m,n+1))
    for j in range(n+1): # col
        Z [:,j] = np.power(x,j)
        # Z[:,j] = x ** j
    a = np.linalg.solve(np.matmul(Z.T,Z),np.matmul(Z.T,y))
    return a
   
# Problem 3
def least_squares_fourier(x, y, k, omega_o):
    sequence = []
    m = len(x)
    n = 2 * k + 1
    assert len(x) == len(y)
    for i in range(n + 1):
        sequence.append(i // 2)
    sequence.pop(0)
    Z = np.zeros((m, n))
    for j in range(n):
        k_iter = sequence[j]
        if j == 0:
            Z[:, j] = 1
        elif j % 2 == 1:
            Z[:, j] = np.cos(omega_o * x * k_iter)
        else:
            Z[:, j] = np.sin(omega_o * x * k_iter)
    a = np.linalg.solve(np.matmul(Z.T, Z), np.matmul(Z.T, y))
    return a


# PROBLEM 4
def my_dft(f):
    n = len(f)
    omega_o = 2 * np.pi / n
    k = np.arange(n).reshape(-1, 1)
    l = np.arange(n)
    matrix = np.exp(-1j * k * omega_o * l)
    F = np.dot(matrix, f)
    return F


# PROBLEM 5, poly interp
def u_cal(u, n):
    temp = u
    for i in range(1, n):
        temp *= (u - i)
    return temp

def fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f

def my_poly_interp(x, fx, xi, ptype):
    n = len(x)
    if ptype == "Lagrange":
        fxi = np.zeros_like(xi)
        for i in range(len(xi)):
            for j in range(n):
                lagrangebasis =  np.prod([(xi[i] - x[m]) / (x[j] - x[m]) for m in range(n) if m != j])
                fxi[i] += fx[j] * lagrangebasis 
        return fxi
    elif ptype == "Newton":
        n = len(x)
        forward_diff_table = np.zeros((n, n))
        forward_diff_table[:, 0] = fx

        for i in range(1, n):
            for j in range(n - i):
                forward_diff_table[j, i] = forward_diff_table[j + 1, i - 1] - forward_diff_table[j, i - 1]

        u = (xi - x[0]) / (x[1] - x[0])
        interpolated_value = forward_diff_table[0, 0]
        for i in range(1, n):
            interpolated_value += (u_cal(u, i) * forward_diff_table[0, i]) / fact(i)

        return interpolated_value
    else:
        return error
