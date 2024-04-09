import numpy as np 


#let's write a function that does naive Gauss elimination
def naive_Gauss(A,b):
    #takes in numpy array A, numpy vector b
    #uses Gauss elimination without pivoting to get solution to Ax=b
    #returns numpy vector x

    nrow,ncol=A.shape
    nentries = len(b)
    #check we have square matrix
    if nrow!=ncol or nentries!=ncol:
        print("Number of rows must equal number of columns or sixe of RHS doesn't match! Exiting")
        return 0
    else:
        #create augmented matrix
        Aug_mat=np.zeros((nrow,ncol+1))
        Aug_mat[:,:ncol]=A
        Aug_mat[:,ncol]=b

        #create empty vector to store solution
        x=np.zeros(ncol)

        #first is process of forward elimination
        #loop from row 0 until row n-1
        for i in range(nrow-1):
            #partial pivoting would go in here

            pivot_elem=Aug_mat[i,i]
            pivot_row = Aug_mat[i,:]
            for j in range(i+1,nrow):
                scaling_factor = -Aug_mat[j,i]/pivot_elem
                #print(scaling_factor)
                #perform a row operation
                Aug_mat[j,:] = Aug_mat[j,:] + scaling_factor*pivot_row
                #print(i,j)
        
        
        #now do backward sub
        #note, first back sub is trivial so inner loop is 0 iterations
        for i in range(nrow-1,-1,-1):
            tmp = Aug_mat[i,-1]
            for j in range(ncol-1,i,-1):
                tmp -= x[j]*Aug_mat[i,j]
            x[i] = tmp/Aug_mat[i,i]
        return x

#a numpy array
# A = np.array( [ [1,-2,1], [2,-1,2], [3,-1,2] ]   )
# #a numpy vector
# b = np.array([8,10,11])

A = np.array( [[1,1,1],[-1,0,-3],[2,-2,1]])
b = np.array([8, -15, 26])
x_naive = naive_Gauss(A,b)

print("Our solution",x_naive)
print("numpy solve solution",np.linalg.solve(A,b))
