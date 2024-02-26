import numpy as np
error = "error"

# PROBLEM 6
def naive_LU(A):
    nrows, ncols = A.shape
    if nrows != ncols:
        # print("Not a square matrix. Cannot compute LU decomposition.")
        return error
    A = A.astype(np.float64)  # Ensure A is of float64 dtype
    L = np.identity(nrows, dtype=np.float64)
    U = np.copy(A)
    for p in range(nrows - 1):
        pivot_elem = U[p, p]
        if pivot_elem == 0:
            # print("Zero pivot encountered. LU decomposition is not possible.")
            return error
        
        for row_below_p in range(p + 1, nrows):
            L[row_below_p, p] = U[row_below_p, p] / pivot_elem
            U[row_below_p, p:] -= L[row_below_p, p] * U[p, p:]
    return L, U

# PROBLEM 7
def solve_LU(L,U,b):
    Lshape = L.shape
    Ushape = U.shape
    if not(Lshape[0]==Lshape[1] and Ushape[0]==Ushape[1] and Lshape[0] == Ushape[1] and len(b) == Ushape[1]):
        #print("wrong matrix input shapes")
        return error
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
    try:
        L,U = naive_LU(A)
    except ValueError:
        #print("not a square matrix or otherwise incorrect input")
        return error
    if np.linalg.matrix_rank(A) !=A.shape[0]:
        #print("not an invertable matrix")
        return error
    maxc = np.shape(A)[0]
    I = np.identity(maxc)
    A_inv = np.zeros((maxc,maxc))
    for c in range (np.shape(A)[0]):
        A_inv[:,c] = solve_LU(L,U,I[:,c])
    return A_inv

# PROBLEM 12
def Richardson_it(A, b, omega, tol, max_it):
    shap = A.shape
    if shap[0] != shap[1]:
        # print("A shape is not square")
        return error
    I = np.identity(A.shape[0])
    iteration_matrix = I - omega * A
    spectral_radius = np.linalg.norm(iteration_matrix, ord=2)
    if spectral_radius >= 1:
        return error
    x = np.zeros(A.shape[1])
    n_it = 0 
    err = tol + 1 # ensure starting error greate than minimum error always
    while (n_it == 0 or err > tol)  and n_it <max_it:
        x_next = x + omega * (b-np.dot(A, x))
        err = np.linalg.norm(np.dot(A, x_next) - b, ord=2)
        x = x_next
        n_it+=1
    return x, n_it, err

# PROBLEM 15
def largest_eig(A, tol, max_it):
    err = tol + 1 # ensure error is initialized to be greater than cut-off
    aShape = A.shape
    if aShape[0] != aShape[1]:
        # print("not a square matrix!")
        return err
    x = np.ones(aShape[0])
    lamda_prev = 1.0
    n_it = 0
    while err > tol and n_it < max_it:
        Ax = np.matmul(A, x)
        eig = max(abs(Ax))
        x = Ax/eig
        err = abs((eig -lamda_prev)/eig)
        lamda_prev = eig
        n_it +=1
    return eig, x, n_it, err 
    

# BONUS, Part 1
def my_Cholesky(A):
    shape = A.shape
    if shape[0] != shape[1]:
        # print("Input is not a square matrix")
        return error
    if not np.allclose(A.T, A):  # Check for symmetry
        # print("Input is a square matrix but not symmetric")
        return error
    n = shape[0]
    U = np.zeros_like(A, dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal elements
                try:
                    U[i, i] = np.sqrt(A[i, i] - np.sum(U[:i, i]**2))
                except:
                    # print("matrix is not positive definite")
                    return error
            else:
                # Non-diagonal elements
                U[i, j] = (A[i, j] - np.sum(U[:i, i] * U[:i, j])) / U[i, i]
    return U

# BONUS, Part 2
def my_Gauss_Siedel(A, b, tol, max_it):
    n = len(b)  # Dimension of the system
    x = np.zeros(n)  # Initial guess for solution
    num_iter = 0
    error_norm = tol + 1 # ensure starting error greate than minimum error always
    while error_norm > tol and num_iter < max_it: 
        x_old = x.copy()  # Store the previous solution
        for i in range(n):
            # Calculate the new value for x[i]
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i+1):], x_old[(i+1):])) / A[i, i]
        # Check for convergence using the error norm
        error_norm = np.linalg.norm((np.dot(A, x) - b), ord = 2)
    num_iter +=1
    return x
