import numpy as np
def naive_LU(A):
    """
    Returns the LU decomposition of a matrix.
    Args:
        matrix A: numpy matrix (Should be square)
    Returns:
        tuple decomp: tuple of the L and U matrixes of the LU decomposition for A
    """
    nrows, ncols = A.shape
    
    if nrows!=ncols:
        print("Not a square matrix. Cannot compute LU decomposition.")
        return 1
    L = np.identity(nrows)
    maxp = nrows-1 # max pivots
    for p in range(maxp):
        pivot_elem = A[p,p]
        for row_below_p in range(p+1, nrows):
            if row_below_p == 2: 
                NEED TO ROW REDUCE A by 1 row/col/step then do.
            L[row_below_p,p] = A[row_below_p,p]/pivot_elem
    print(L)
A = np.matrix([[4,2,2],[0,5,4],[2,7,9]])
naive_LU(A)