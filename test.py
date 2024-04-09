import numpy as np

A = np.array([[4, 7], 
              [1, 2], 
              [5, 6]])

B = np.array([[4, 3, 7], 
              [1, 2, 7], 
              [2, 0, 4]])

C = np.array([[3], 
              [6], 
              [1]])

D = np.array([[9, 4, 3, -6],
              [2,-1 ,7 ,5]])

E = np.array([[1 ,5 ,8],
              [7 ,2 ,3],
              [4 ,0 ,6]])

F = np.array([[3 ,0 ,1],
              [1 ,7 ,3]])

G = np.array([[7],[6],[4]])
import numpy as np

# Assuming A, B, C, D, E, F are predefined numpy arrays

# (1) E + B
result1 = E + B

# (2) A + F
# result2 = A + F

# (3) B - E
result3 = B - E

# (4) 7 * B
result4 = 7 * B

# (5) Transpose of C
result5 = C.T

# (6) Element-wise multiplication of E and B
result6 = E @B

# (7) Matrix multiplication of B and A
result7 = np.dot(B, A)

# (8) Transpose of D
result8 = D.T

# (9) Element-wise multiplication of A and C
result9 = A * C

# (10) Matrix multiplication of I and B
I = np.eye(B.shape[0])
result10 = np.dot(I, B)

# (11) Element-wise multiplication of transpose of E and E
result11 = E.T @ E

# (12) Matrix multiplication of transpose of C and C
result12 = np.dot(C.T, C)
print("Result 1: \n", result1)
# print("Result 2: \n", result2)
print("Result 3: \n", result3)
print("Result 4: \n", result4)
print("Result 5: \n", result5)
print("Result 6: \n", result6)
print("Result 7: \n", result7)
print("Result 8: \n", result8)
print("Result 9: \n", result9)
print("Result 10: \n", result10)
print("Result 11: \n", result11)
print("Result 12: \n", result12)
print ("***********************************")

A = np.array([[6,-1],[12,8],[-5,4]])
B = np.array([[4,0],[.5,2]])
C = np.array([[2,-2],[3,1]])

print("A X B \n" , A@B)
print("A X C \n" , A@C)
print("B X C \n" , B@C)
print("C X B \n" , C@B)

print(np.linalg.det(np.array([[0,-3,7],[1,2,-1],[5,-2,0]])))
A = np.array([[0,-3,7],[1,2,-1],[5,-2,0]])
B=np.array([[4,0,3]]).T

print(np.linalg.solve(A,B))

A = np.array([[10,2,-1],[-3,-5,2],[1,1,6]])
B=np.array([[27,-61.5,-21.5]]).T

print(np.linalg.solve(A,B))

A = np.array([[2,-6,-1],[-3,-1,7],[-8,1,-2]])
B=np.array([[-38,-34,-20]]).T

print(np.linalg.solve(A,B))
print(np.linalg.det(A))