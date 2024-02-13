import numpy as np

def solve_LU(L, U, b):
    """
    Solves a linear system Ax=b given the LU decomposition of A.
    Args:
        L: numpy array, lower triangular matrix of the LU decomposition of A
        U: numpy array, upper triangular matrix of the LU decomposition of A
        b: numpy array, right hand side vector of the system Ax=b
    Returns:
        x: numpy array, solution to the original system Ax=b
    """
    # Solve Ly = b using forward substitution
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Solve Ux = y using backward substitution
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x

# Test the function
L = np.array([[1, 0, 0], [2/3, 1, 0], [-1/3, -4/11, 1]])
U = np.array([[3, -2, 1], [0, -22/3, -14/3], [0, 0, 40/11]])
b = np.array([-10,44,-26])
print(solve_LU(L,U,b))
print(np.linalg.solve(np.matmul(L,U),b))
print("l",L)