import numpy as np
A = np.array([[2, 8, 10], [8, 4, 5],[10, 5, 7]])
minvect = np.linalg.eig(A)[1][:,2]
print(minvect/max(abs(minvect)))