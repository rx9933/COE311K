# Matrix decomposition; gauss demo
import numpy as np

def naive_gauss(A, b):
    """
    Uses Naive Gauss elimination without pivoting to get solution to Ax = b.
    Args:
        array A: coefficient array.
        vector b: values of A*x.
    Returns:
        vector x: the solution.
    """
    nrow, ncol = A.shape
    nentries = len(b)
    if nrow!=ncol or nentries!=ncol:
        print("Must be square matrix or size of RHS does not match! Exiting function.")
        return 0
    # forward elimination; loop from row 0 to row nrow-1 where nrow is max row
    aug_mat = np.zeros((nrow, ncol+1))
    aug_mat[:,:ncol] = A
    aug_mat[:,ncol] = b
    # aug_mat = np.vstack((A,b))
    for i in range(nrow-1):
        # partial pivoting
        pivot_elem = aug_mat[i,i]
        pivot_row = aug_mat[i,:]
        for j in range(i+1,nrow): # all bottom rows
            scaling_factor = -aug_mat[j,i]/pivot_elem
            aug_mat[j,:]  = aug_mat[j,:] + scaling_factor*pivot_row
    # return aug_mat
    # now backward substitution
    # for i in range(nrow-1, 0, -1):
        # aug_mat aug_mat[i,ncol-1]
A = np.array([[1,-2,1],[2, -1,2],[3,-1,2]])
b = np.array([8,10,11])

naive_gauss(A,b)