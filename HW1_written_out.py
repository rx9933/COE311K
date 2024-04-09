from matplotlib import pyplot as plt
from math import sqrt, tanh, factorial, pi
import numpy as np
from typing import List
# problem 1
def exact_velocity(c_d:float, m:float, t:List[float], g:float)->list:
    """
    Calculates the exact velocity of an object of mass m at the times listed in t. 
    Uses the solution to the differential equation (DE) model of the velocity of a falling object subject to the forces of gravity and wind drag.
    All units are assumed to be of the same scale (i.e, SI units are assumed).
    Args:
        float c_d: drag coefficient
        float m: mass of the object 
        vector/List[float] t: a vector of times the user wants to compute the velocity at
        float g: gravitational constant 
    Returns:
        vector[float] v: the velocity at times from the input vector t according to the exact solution to the DE model 
    """
    v = np.zeros(len(t)) # initialize velocity
    for timeindex, timevalue in enumerate(t): # loop over each time 
        v[timeindex] = sqrt(g*m/c_d)*tanh(sqrt(g*c_d/m)*timevalue) 
    return np.array(v)

# problem 3
def forward_Euler_velocity(c_d:float, m:float, t:List[float], g:float)->list:
    """
    Uses forward euler approximation to approximate the time deravative \
        and solve for the velocity of a falling object at the times in t.
        All units are assumed to be of the same scale (i.e, SI units are assumed).
    Args:
        float c_d: drag coefficient
        float m: mass of the object 
        vector/List[float] t: a vector of times the user wants to compute the velocity at
        float g: gravitational constant 
    Returns:
        vector[float] v: the approximated velocity at times from the input vector t 
    """
    numSteps = len(t)
    v = np.zeros(numSteps)
    stepNo = 1
    while stepNo < numSteps: # loop through each time step
        dt = t[stepNo]-t[stepNo-1] # delta t
        v[stepNo] = v[stepNo-1] + dt*(g-c_d/m*(v[stepNo-1])**2) # approx of v'
        stepNo+=1
    return np.array(v)

# problem 5
def root_mean_square_error(maxt:float, dt:float, c_d:float, m:float, g:float) -> float:
    """
    Calculates the root mean square error of the velocity at any given time point.
    Args:
        float maxt: the maximum time to be calculated.
        float dt: the step size or delta time interval between velocity re-calculations.
        float c_d: drag coefficient
        float m: mass of the object 
        float g: gravitational constant
    Returns:
        float rmse: the root mean square error of the velocities of the data sample. The data sample is defined\
        as all the times between 0 and maxt with the delta t as dt.         
    """ 
    tsteps = int(maxt/dt) + 1 # the total number of time instances where to calculate velocity
    t = np.linspace(0,maxt, tsteps) # each time spot
    # the difference between the exact and approximated velocities at the times listed in t
    dv = np.array(exact_velocity(c_d, m, t, g) - forward_Euler_velocity(c_d, m, t, g)) 
    rmse = sqrt(np.sum(dv**2)/tsteps) # compute the quadrature of dv over the number of time steps
    return rmse

# bonus, part1
def mat_mat_mul(A,B):
    """
    Performs the matrix multiplication betwen two matrixes.
    Args:
        matrix A: the first matrix
        matrix B: the second matrix
    Returns:
        matrix result: the product of matrix1 and matrix2
        or, int 1: an error has occured
    """
    # Check that dimensions match (num cols of first matrix = num rows of second matrix)
    if np.shape(A)[1]!= np.shape(B)[0] :
        print("Incorrect matrix sizes. Cannot multiply.")
        return 1

    # Initialize result matrix with zeros
    result = np.zeros((A.shape[0], B.shape[1])) # shape should be rows of first matrix and columns of second matrix
    # Perform matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def approximate_sin(x:float, es:float, maxit:int) -> float:
    """
    Function uses the Maclaurin series to estimate sin(x), where x is a particular value in radians.
    Args: 
        float x: point at which to approximate sin(x), in radians
        float es: the relative error allowed
        float maxit: the maximum number of iterations allowed to calculate sin(x)
    Returns
        float approximation: the approximated value of sin(x)
    """
    # Initialize variables
    approximation = 0
    old_approximation = 0
    relerror = 100
    iter = 0
    # Loop until the maximum number of iterations or until the error is less than the allowed error
    while relerror > es and iter < maxit: 
        old_approximation = approximation # update previous approximation
        approximation += ((-1)**iter) * (x**(2*iter+1)) / factorial(2*iter+1) # use taylor series approximation
        if abs(approximation) > 10**-7: # prevent any division by 0.0 (or close value)
            relerror = abs((approximation - old_approximation) / approximation)
        iter += 1 # update iteration number
    return approximation


# uncomment following lines for test cases





# # bonus, part2
print("approximate sin value:", approximate_sin(pi/4, .0001, 10))
mat_mat_mul(np.array([[1,2,3],[2,3,4]]), np.array([[2,2], [1,1]]))