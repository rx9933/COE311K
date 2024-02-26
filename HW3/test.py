import numpy as np

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

# Example usage
x = np.linspace(0, 10, 11)
alpha = [-4, 2, 2]
k = 2 # originally 1
omega_o = 1
# y = alpha[0] + alpha[1] * np.cos(x) + alpha[2] * np.sin(x) + np.random.randn(len(x))
y = -7+2*np.cos(x) -3*np.sin(x) + np.cos(2*x) + .5*np.sin(2*x) + np.random.randn(len(x))
print(least_squares_fourier(x, y, k, omega_o))
