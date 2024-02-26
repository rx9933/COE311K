'''
# PROBLEM 1
L = np.array([[1, 0, 0], [2/3, 1, 0], [-1/3, -4/11, 1]])
U = np.array([[3, -2, 1], [0, 22/3, -14/3], [0, 0, 40/11]]) 
b = np.array([-10,44,-26])
print(solve_LU(L,U,b))
print(np.linalg.solve(np.matmul(L,U),b))


# PROBLEM 2
A= np.array([[8,2,1],[3,7,2],[2,3,9]])
L,U = naive_LU(A)
print(L)
print(U)

# PROBLEM 3
A = np.array([[10,2,-1],[-3,-6,2],[1,1,5]])
L,U = naive_LU(A)

# print(L)
# print(U)
# print(np.matmul(L,U))
# print(np.matmul(np.array([[1,0,0],[-3/10,1,0],[1/10,-4/27,1]]), np.array([[10,2,-1],[0,-5.4,1.7],[0,0,289/54]])))
# print(np.matmul(np.array([[1,0,0],[-3/10,1,0],[1/10,-4/27,1]]), np.array([[10,2,-1],[0,-5.4,1.7],[0,0,289/54]])))

# PROBLEM 4
# b = np.array([27,-61.5,-21.5])
# print(solve_LU(L,U, b))
# print("A", np.linalg.solve(A,b))

b = np.array([12,18,-6])
# print(solve_LU(L,U, b))
print("A", np.linalg.solve(A,b))

# PROBLEM 5
A = np.array([[2,-6,-1],[-3,-1,7],[-8,1,-2]])
b = np.array([-38,-34,-40])

print(np.linalg.solve(A,b))
# p,l,u = lu(A)
# print(np.linalg.inv(p)@A)
# print(l@u)
P = np.array([[0,0,1],[1,0,0],[0,1,0]])
L = np.array([[1,0,0],[-1/4,1,0],[3/8,11/46,1]])
U = np.array([[-8,1,-2], [0, -23/4, -3/2], [0,0,373/46]])
print(np.matmul(P,A))
print(np.matmul(L,U))


# p2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
# l1 = np.array([[1, 0, 0], [-3/8, 1, 0], [1/4, 0, 1]])
# l2 = np.array ([[1, 0, 0], [0, 1, 0], [0, -11/46, 0]])

# a1 = np.matmul(p2,np.linalg.inv(l1))
# a2 = np.matmul(a1, np.linalg.inv(p2))
# a3 = np.matmul (a2, np.linalg.inv(l2))

# PROBLEM 6
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
l,u =naive_LU(A)
print(np.matmul(l,u))


# PROBLEM 7

L = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]])
U = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
b = np.array([1,2,3])

A = np.matmul(L,U)
# print(A)
d =np.linalg.solve(A, b)
# print(np.linalg.solve(U, d))

solve_LU(L,U,b)
L = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]])
U = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
print(solve_LU(L,U,b))
print(np.linalg.solve(np.matmul(L,U),b))

# PROBLEM 8
F = np.array([[10, 2, - 1], [-3, -6, 2], [1,1,5]])
b = np.array([27, -61.5, -21.5])

print(np.linalg.inv(F))
print(np.linalg.solve(F, b))


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
print(np.linalg.solve(A,b))

# PROBLEM 10

A = np.array([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
 
A = np.array([[2,7], [3,4], [6,5]])

A = np.array([[5,4,3], [11, 10, 8]])
print(np.linalg.norm(A))
print(np.linalg.norm(A, ord=1))
print(np.linalg.norm(A, ord=np.inf))

# PROBLEM 11
A = np.array([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
Ainv = inv_using_naive_LU(A) 
# print(Ainv)
# print(np.linalg.inv(A))
assert(np.allclose(Ainv, np.linalg.inv(A)))

# PROBLEM 12
A = np.array([[8, 2, -10], [-9, 1, 3], [15, -1, 6]])
b = np.array([-2, 3,8])
w = .01
I = np.identity(A.shape[1])

A = np.random.rand(3, 3)
w = 1
I = np.eye(3)
iteration_matrix = I - w * A
spectral_radius = np.linalg.norm(iteration_matrix, ord=2)
while spectral_radius <= 1:
    A = np.random.rand(3, 3)
    iteration_matrix = I - w * A
    spectral_radius = np.linalg.norm(iteration_matrix, ord=2)
# print("Matrix A:")
# print(A)
# print(np.linalg.norm(I-w*A, ord =  2))

print(Richardson_it(A, b, w, .1, 890))
# print(np.linalg.solve(A,b))



# PROBLEM 13
A = np.array([[2,8,10], [8, 4, 5], [10, 5, 7]])
print(largest_eig(A, .004, 3))

# PROBLEM 14
A = np.array([[2, 8, 10], [8, 4, 5], [10, 5, 7]], dtype=float)
# print("inv", np.linalg.inv(A))
print("AA",largest_eig(np.linalg.inv(A), .000001, 30))
print(np.linalg.eig(A))


# PROBLEM 15
A = np.array([[40, -20, 0], [-20, 40, -20], [0, -20, 40]])
print(largest_eig(A, .000000001, 30))
print("A",np.linalg.eig(A))



# BONUS p.1
# A = np.array([[4, 12, -16],[12, 37, -43], [-16, -43, 98]])
A = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]])
print(my_Cholesky(A))
print(np.linalg.cholesky(A).T) # RETURNS u
'''

# BONUS part 2
nit=100
e = .001
A = np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]])
b = np.array([-2, 3,8])
print(my_GaussSiedel(A, b, e, nit))
print(np.linalg.solve(A,b))

'''


# A = np.array([[3, -2, 1],[2, 6, -4],[-1, -2, 5]])
# b = [-10, 44, -26]
# L,U = naive_LU(A)
# print(solve_LU(L,U, b))
'''