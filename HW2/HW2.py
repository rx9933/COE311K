import numpy as np
from scipy.linalg import lu
from math import sqrt
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
    return x

# BONUS, Part 1
# def my_Cholesky(A):
#     shape = A.shape
#     if shape[0]!=shape[1]:
#         print("input is not a square matrix")
#         return 1
#     if A.T.any() != A.any(): # is symmetric
#         print("input is a square but not a symmetric matrix")
#         return 1
    # U = np.zeros(shape)
    # for i in range(shape[0]):
    #     for j in range(i+1, shape[0]):
    #         sum = 0
    #         for k in range(i-1):
    #             sum += U[k,i] * U[k,j]
    #         U[i,j] = A[i,j] - sum
    # for i in range(shape[0]): # each row
    #     # along diagonal
    #     ukisq = 0
    #     for k in range(1, i-1):
    #         ukisq += U[k,i]**2
    #     U[i,i] = sqrt(A[i,i]-ukisq)
    # print(U.T)
def my_Cholesky(A):
    shape = A.shape
    if shape[0] != shape[1]:
        print("Input is not a square matrix")
        return None
    if not np.allclose(A.T, A):  # Check for symmetry
        print("Input is a square matrix but not symmetric")
        return None
    n = shape[0]
  
    U = np.zeros_like(A, dtype=np.float64)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal elements
                U[i, i] = np.sqrt(A[i, i] - np.sum(U[:i, i]**2))
            else:
                # Non-diagonal elements
                U[i, j] = (A[i, j] - np.sum(U[:i, i] * U[:i, j])) / U[i, i]
    return U
    

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
"""
# PROBLEM 1
L = np.array([[1, 0, 0], [2/3, 1, 0], [-1/3, -4/11, 1]])
U = np.array([[3, -2, 1], [0, -22/3, -14/3], [0, 0, 40/11]]) 
b = np.array([-10,44,-26])
print(solve_LU(L,U,b))
print(np.linalg.solve(np.matmul(L,U),b))

# PROBLEM 2
A= np.array([[8,2,1],[3,7,2],[2,3,9]])
L,U = naive_LU(A)
print(L)
"""
# PROBLEM 3
A = np.array([[10,2,-1],[-3,-6,2],[1,1,5]])
L,U = naive_LU(A)

# print(L)
# print(U)
# print(np.matmul(np.array([[1,0,0],[-3/10,1,0],[1/10,-4/27,1]]), np.array([[10,2,-1],[0,-5.4,1.7],[0,0,289/54]])))
# print(np.matmul(np.array([[1,0,0],[-3/10,1,0],[1/10,-4/27,1]]), np.array([[10,2,-1],[0,-5.4,1.7],[0,0,289/54]])))

# PROBLEM 4
# b = np.array([27,-61.5,-21.5])
# print(solve_LU(L,U, b))

# b = np.array([12,18,-6])
# print(solve_LU(L,U, b))

# PROBLEM 5
A = np.array([[2,-6,-1],[-3,-1,7],[-8,1,-2]])
b = np.array([-38,-34,-40])

# print(np.linalg.solve(A,b))
# p,l,u = lu(A)
# print(np.linalg.inv(p)@A)
# print(l@u)
P = np.array([[0,0,1],[1,0,0],[0,1,0]])
L = np.array([[1,0,0],[-1/4,1,0],[3/8,11/46,1]])
U = np.array([[-8,1,-2], [0, -23/4, -3/2], [0,0,373/46]])
# print(np.matmul(P,A))
# print(np.matmul(L,U))

# p2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
# l1 = np.array([[1, 0, 0], [-3/8, 1, 0], [1/4, 0, 1]])
# l2 = np.array ([[1, 0, 0], [0, 1, 0], [0, -11/46, 0]])

# a1 = np.matmul(p2,np.linalg.inv(l1))
# a2 = np.matmul(a1, np.linalg.inv(p2))
# a3 = np.matmul (a2, np.linalg.inv(l2))

# BONUS p.1
# A = np.array([[4, 12, -16],[12, 37, -43], [-16, -43, 98]])
A = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]])
print(my_Cholesky(A))
print(np.linalg.cholesky(A).T) # RETURNS L
