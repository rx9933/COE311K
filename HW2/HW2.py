import numpy as np
from scipy.linalg import lu

# PROBLEM 6
def naive_LU(A):
    """
    Returns the LU decomposition of a matrix.
    Args:
        A: numpy matrix (Should be square)
    Returns:
        tuple decomp: tuple of the L and U matrices of the LU decomposition for A
    """
    nrows, ncols = A.shape
    
    if nrows != ncols:
        print("Not a square matrix. Cannot compute LU decomposition.")
        return None, None
    
    A = A.astype(np.float64)  # Ensure A is of float64 dtype
    L = np.identity(nrows, dtype=np.float64)
    U = np.copy(A)
    for p in range(nrows - 1):
        pivot_elem = U[p, p]
        if pivot_elem == 0:
            print("Zero pivot encountered. LU decomposition is not possible.")
            return None, None
        
        for row_below_p in range(p + 1, nrows):
            L[row_below_p, p] = U[row_below_p, p] / pivot_elem
            U[row_below_p, p:] -= L[row_below_p, p] * U[p, p:]
    return L, U

# PROBLEM 7
def solve_LU(L,U,b):
    # solve Ld = b for b
    d = np.zeros(len(b))
    x = np.copy(d)
    for r in range(len(b)): # forward sub
        d[r] = b[r]-np.dot(L[r,:r],d[:r])
    # solve Ux = d for x
    nrows = len(b)
    for r in range(nrows-1, -1, -1):
        x[r] = (d[r] - np.dot(U[r,r+1:], x[r+1:]))/U[r,r]
    print("x", x)
    return x

# PROBLEM 7
"""
L = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]])
U = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
b = np.array([1,2,3])

A = np.matmul(L,U)
print(A)
d =np.linalg.solve(A, b)
print(np.linalg.solve(U, d))

solve_LU(L,U,b)
"""
# PROBLEM 1
L = np.array([[1, 0, 0], [2/3, 1, 0], [-1/3, -4/11, 1]])
U = np.array([[3, -2, 1], [0, -22/3, -14/3], [0, 0, 40/11]]) b 
b = np.array([-10,44,-26])
print(solve_LU(L,U,b))
print(np.linalg.solve(np.matmul(L,U),b))

# PROBLEM 2
A= np.array([[8,2,1],[3,7,2],[2,3,9]])
L,U = naive_LU(A)
print(L)
