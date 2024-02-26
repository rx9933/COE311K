# PROBLEM 1
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, e
error = "error"
x = np.array([.1, .2, .4, .6, .9, 1.3, 1.5, 1.7, 1.8])
m = len(x)
y = np.array([.75, 1.25, 1.45, 1.25, .85, .55, .35, .28, .18])

A = np.zeros((2,2))
b = np.zeros(2)

A[0,0] = m
A[0,1] = np.sum(x)
A[1,0] = np.sum(x)
A[1,1] = np.sum(x**2)

b[0] = np.sum(np.log(y)-np.log(x))
b[1] = np.sum(np.log(y)*x-np.log(x)*x)


a = np.linalg.solve(A,b)
constants = [e**(a[0]),a[1]]
yapprox = constants[0]*x*e**(constants[1]*x)

plt.plot(x,yapprox, "r", label = "least squares fit")
plt.scatter(x,y)
plt.title("Least Squares Fit")
plt.legend(["least squares fit, y = 9.662xe^(-2.473x)","data points"])
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


def least_squares_poly(x,y,k):
    m = len(x)
    n = k
    assert(len(x) == len(y))

    Z = np.zeros((m,n))
    for j in range(n): # col
        Z [:,j] = x**j
    a = np.linalg.solve(np.matmul(Z.T,Z),np.matmul(Z.T,y))
    return a   

def least_squares_fourier(x, y, k, omega_o):
    sequence = []
    m = len(x)
    n = 2 * k + 1
    assert len(x) == len(y)

    for i in range(n + 1):
        sequence.append(i // 2)
    sequence.pop(0)
    Z = np.zeros((m, n))
    for j in range(n):
        k_iter = sequence[j]
        if j == 0:
            Z[:, j] = 1
        elif j % 2 == 1:
            Z[:, j] = np.cos(omega_o * x * k_iter)
        else:
            Z[:, j] = np.sin(omega_o * x * k_iter)
    
    a = np.linalg.solve(np.matmul(Z.T, Z), np.matmul(Z.T, y))

    return a

########################
# Problem 2#
#######################
m = 100
x = np.linspace(0,10, m)

alpha = [1,23]

# generate some noisy data
y = alpha[0] + x*alpha[1] + np.random.randn(m)
print(least_squares_poly(x,y,2))

##########
#problem 3
#######
x = np.linspace(0,10,100)
m = len(x)
alpha = [33,2,3]
k =1
omega_o = 1

y = alpha[0] + alpha[1] * np.cos(x) + alpha[2] * np.sin(x) + np.random.randn(m) 
print(least_squares_fourier(x, y, k, omega_o))

################
## problem 3
############
x = np.linspace(0, 10, 11)
# alpha = [-4, 2, 2]
k = 2 # originally 1
omega_o = 1
# y = alpha[0] + alpha[1] * np.cos(x) + alpha[2] * np.sin(x) + np.random.randn(len(x))
y = -7+2*np.cos(x) -3*np.sin(x) + np.cos(2*x) + .5*np.sin(2*x) + np.random.randn(len(x))
print(least_squares_fourier(x, y, k, omega_o))

