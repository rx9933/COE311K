import numpy as np
error="error"

# Problem 1
def my_linear_interpolation(x,fx,xi):
    '''
    Takes in x - input data
    fx - data you want to fit curve to, should be same length as x
    xi - the points you wish to interpolate at

    return fxi, a vector same size as xi with interpolated values using linear interpolation
	
	return error if any points in xi lie outside the range of x
    '''
    if max(xi) > max(x) or min(xi) < min(x):
          return error

    x_sorted, fx_sorted = zip(*sorted(zip(x, fx)))

    # Find the values of a that are above and below xi
    x2 = max(filter(lambda x: x <= xi, x_sorted))
    x1 = min(filter(lambda x: x > xi, x_sorted))

    # Get the corresponding values of b
    y2 = fx_sorted[x_sorted.index(x2)]
    y1 = fx_sorted[x_sorted.index(x1)]
   
    slope = (y2-y1)/(x2-x1)
    dx = xi-x1
    
    yi = y1 + slope*dx
    return yi

# Problem 1 Test
x = [2, 3, 45, 1]
fx = [4, 10, 90, 2]
# print(my_linear_interpolation(x,fx,7))

def my_cubic_spline_interpolation(x,fx,xi):
    '''
    Takes in x - input data
    fx - data you want to fit curve to, should be same length as x
    xi - the points you wish to interpolate at

    return fxi, a vector same size as xi with interpolated values using cubic splines

    return error if any points in xi lie outside the range of x
    '''

    if max(xi) > max(x) or min(xi) < min(x):
        return error 
    n = len(x)
    n_s = n - 1 # number of splines
    # first sort x, fx to ensure increasing order of x
    x, fx = zip(*sorted(zip(x, fx)))
    # set up each spline as Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    # must solve for 4 coefficients per spline (n-1 splines) = 4(n-1) equations
    # Plan: with 6 equations (as described below), set up a system of 3(n-1) x 3(n-1) equations 
    # (Equation 1, which directly solves for ai, is substituted first). 
    # A*r = s, where A is matrix of order 3(n-1), r is a vector of length 3(n-1), and s is a vector of length 3(n-1).
    # Solve for r to find remaining coefficients.

    A = np.zeros((3*n_s, 3*n_s))
    r = [0]*3*n_s
    s = r.copy()

    # Note, r has form [b1,c1,d1,b2,c2,d2,....bn-1, cn-1, dn-1].T (Transpose)

    # Equation 1, each spline must pass through the data points.
    # ai = yi for i = [1,n-1] 
    # n-1 equations
    a = fx[:-1]

    # Equation 2, equate right value of left spline with left value of right spline.
    # y[i+1]-y[i] = (x[i+1]-x[i]) * b[i] + (x[i+1]-x[i])^2 * c[i] + (x[i+1]-x[i])^3 * d[i] for i = [1, n-1]
    # n-1 

    # set up A matrix first
    for row in range(n-1):
        A[row, 3*row:3*(row+1)] = [x[row+1] - x[row], (x[row+1] - x[row])**2, (x[row+1] - x[row])**3]
    # A*r = s, set up left side of equation
    s[:n-1] = np.subtract(fx[1:], fx[:-1])
 
    # Equation 3, equate slopes before and after a given point.
    #  b[i] - b[i+1] + (x[i+1]-x[i]) * 2c[i] + (x[i+1]-x[i])^2 * 3d[i] = 0 for i = [1, n-2]
    # n-2 equations
    
    # set up A matrix first
    for row in range(n-2):
        A[n-1 + row, 3*row:3*row+4] = [1, 2 * (x[row+1] - x[row]), 3 * (x[row+1] - x[row])**2, -1]
    # A*r = s, set up left side of equation
    s[n:2*n-2] = np.zeros(n-2)
  
    # Equation 4, equate second deravatives of splines before and after a given point. 
    # 2c[i] - 2c[i+1] + (x[i+1]-x[i]) * 6d[i] = 0 for i = [1, n-2]
    # n-2 equations

    # set up A matrix first
    for row in range(n-2):
        A[2*n-3 + row, 1+3*row:1+3*row+4] = [2, 6 * (x[row+1] - x[row]), 0, -2]
    # A*r = s, set up left side of equation
    s[2*n-2:3*n-4] = np.zeros(n-2)
      

    # Equation 5, the second deravative of the start of the first spline is 0. For natural look, assumed/set value.
    # c[1] = 0
    # 1 equation
    A[3*n-5,1] = 1
    s[3*n-4] = 0

    # Equation 6, the second deravative of the end of the last spline is 0. For natural look, assumed/set value.
    # 2c[n-1]+6d[n-1] (x[n]-x[n-1]) = 0
    # 1 equation
    
    A[3*n-4,3*n-5:3*(n-1)] = [2, 6*(x[n-1]-x[n-2])]
    s[3*n-4] = 0

    # Solve for r
    r = np.linalg.solve(A,s)
    # print(r)
       # Find the values of a that are above and below xi
    fxi = []
    for interp in xi:
        x1 = max(filter(lambda x: x <= interp, x))
        x2 = min(filter(lambda x: x > interp, x))

        # Get the corresponding values of b
        y1 = fx[x.index(x1)]
        y2 = fx[x.index(x2)]
        i= np.where(np.isclose(x, x1)) # array of index of lower x point
        i = int(max(i))

        # spline number i+1 
        val = a[i] +  r[3*i] * (interp-x1) + r[3*i+1] * (interp-x1)**2 + r[3*i+2] * (interp-x1)**3
        fxi+=[val]
    return fxi



x = [1, 2, 3, 4]
fx = [1, 8, 27, 64]
xi = [1.2, 2.4, 2.5]    

print("calculated value", my_cubic_spline_interpolation(x,fx,xi))
from scipy.interpolate import CubicSpline
import scipy.interpolate
cs = CubicSpline(x, fx)
print("interp val, splined scipy",cs(xi))
print("actual", np.array(xi)**3)




def my_bisection_method(f,a,b,tol,maxit):
	'''
    Takes in f - a possibly nonlinear function of a single variable
    a - the lower bound of the search interval
	b - the upper bound of the search interval
	tol - allowed relative tolerance
	maxit - allowed number of iteraions

    return root,fx,ea,nit
	root - the final estimate for location of the root
	fx - the estimated values for f at the root (will be near 0 if we are close to the root)
	ea - the final relative error
	nit - the number of iterations taken
	
	return error if the sign of f(a) is the same as the sign of f(b) since this means it is possible
	that a root doesnt lie in the interval
    '''
	return root,fx,ea,nit

def modified_secant_method(f,x0,tol,maxit,delta):
    '''
	Takes in f - a possibly nonlinear function of a single variable
    x0 - the initial guess for the root
	tol - allowed relative tolerance
	maxit - allowed number of iteraions
	delta - the size of the finite difference approximation

    return root,fx,ea,nit
	root - the final estimate for location of the root
	fx - the estimated values for f at the root (will be near 0 if we are close to the root)
	ea - the final relative error
	nit - the number of iterations taken
	
	no error checking is necessary in this case
	'''
