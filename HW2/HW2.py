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

# PROBLEM 11
def inv_using_naive_LU(A):
    L,U = naive_LU(A)
    maxc = np.shape(A)[0]
    I = np.identity(maxc)
    Ainv = np.zeros((maxc,maxc))
    for c in range (np.shape(A)[0]):
        Ainv[:,c] = solve_LU(L,U,I[:,c])
    return Ainv

# PROBLEM 12
def Richardson_it(A, b, w, e, max_it):
    x_current = np.zeros(A.shape[1])
    num_it = 0 
    error = e + 1 # ensure starting error greate than minimum error always
    while (num_it == 0 or error > e)  and num_it <max_it:
        x_next = x_current + w * (b-np.dot(A, x_current))
        error = np.linalg.norm(np.dot(A, x_current) - b)
        x_current = x_next
        num_it+=1
    return x_current

# BONUS, Part 1
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

# BONUS, Part 2
def my_GaussSiedel(A, b, e, nit):
    n = len(b)  # Dimension of the system
    x = np.zeros(n)  # Initial guess for solution
    num_iter = 0
    error_norm = e + 1 # ensure starting error greate than minimum error always
    while error_norm > e and num_iter < nit: 
        x_old = x.copy()  # Store the previous solution

        for i in range(n):
            # Calculate the new value for x[i]
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i+1):], x_old[(i+1):])) / A[i, i]

        # Check for convergence using the error norm
        error_norm = np.linalg.norm((np.dot(A, x_old) - b), ord = 2)

    num_iter +=1

    return x

# PROBLEM 15
def largest_eig(A, e, max_it):
    error = e + 1 # ensure error is initialized to be greater than cut-off
    aShape = A.shape
    if aShape[0] != aShape[1]:
        print("not a square matrix!")
        return None
    x = np.ones(aShape[0])
    lamda_prev = 1.0
    n_it = 0
    while error > e and n_it < max_it:
        Ax = np.matmul(A, x)
        lamda = max(abs(Ax))

        x = Ax/lamda
        print("x",x)
        error = abs((lamda -lamda_prev)/lamda)
        lamda_prev = lamda
        n_it +=1
    return lamda, x, n_it, error 
    

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
b = np.array([27,-61.5,-21.5])
# print(solve_LU(L,U, b))
# print("A", np.linalg.solve(A,b))

b = np.array([12,18,-6])
# print(solve_LU(L,U, b))
# print("A", np.linalg.solve(A,b))

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
# print(my_Cholesky(A))
# print(np.linalg.cholesky(A).T) # RETURNS L

# PROBLEM 8
F = np.array([[10, 2, - 1], [-3, -6, 2], [1,1,5]])
b = np.array([27, -61.5, -21.5])
# print(np.linalg.inv(F))
# print(np.linalg.solve(F, b))

# PROBLEM 9
L = np.array([[1,0,0],[-.25, 1, 0], [.375, 11/46, 1]])
U = np.array([[-8, 1, -2], [0, -5.75, -1.5], [0, 0, 373/46]])
# print(np.matmul(L,U))
A = np.array([[-8, 1, -2], [2, -6, -1], [-3, -1, 7]])
p,l,u = lu(A)
# print(l)
# print(u)
# print(np.linalg.inv(A))
b = np.array([-20, -38, -34])
# print(np.linalg.solve(A,b))

# PROBLEM 10
A = np.array([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
 
A = np.array([[2,7], [3,4], [6,5]])

A = np.array([[5,4,3], [11, 10, 8]])
# print(np.linalg.norm(A))
# print(np.linalg.norm(A, ord=1))
# print(np.linalg.norm(A, ord=np.inf))

# PROBLEM 11
A = np.array([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
Ainv = inv_using_naive_LU(A) 
# print(Ainv)
# print(np.linalg.inv(A))
assert(np.allclose(Ainv, np.linalg.inv(A)))

# PROBLEM 12
b = np.array([-2, 3,8])
w = .01
I = np.identity(A.shape[1])

A = np.random.rand(3, 3)
w = 1
I = np.eye(3)
iteration_matrix = I - w * A
spectral_radius = np.linalg.norm(iteration_matrix, ord=2)
while spectral_radius >= 1:
    A = np.random.rand(3, 3)
    iteration_matrix = I - w * A
    spectral_radius = np.linalg.norm(iteration_matrix, ord=2)
# print("Matrix A:")
# print(A)
# print(np.linalg.norm(I-w*A, ord =  2))

# print(Richardson_it(A, b, w, .1, 890))
# print(np.linalg.solve(A,b))



# def largest_eig(A, e, max_it):
#     return lamda, x, n_it, error 
# PROBLEM 13
A = np.array([[2,8,10], [8, 4, 5], [10, 5, 7]])
# print(largest_eig(A, .004, 3))

# PROBLEM 14
# print(np.linalg.eig(A))
# PROBLEM 15
A = np.array([[40, -20, 0], [-20, 40, -20], [0, -20, 40]])
# print(largest_eig(A, .00001, 30))
# print("A",np.linalg.eig(A))



# PROBLEM 14
A = np.array([[2, 8, 10], [8, 4, 5], [10, 5, 7]], dtype=float)
# print("inv", np.linalg.inv(A))
# print("AA",largest_eig(np.linalg.inv(A), .000001, 30))
# print(np.linalg.eig(A))

# BONUS part 2
nit=100
e = .1
# print(my_GaussSiedel(A, b, e, nit))
# print(np.linalg.solve(A,b))


A = np.array([[3, -2, 1],[2, 6, -4],[-1, -2, 5]])
b = [-10, 44, -26]
L,U = naive_LU(A)
print(solve_LU(L,U, b))