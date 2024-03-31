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
    # check if any points in xi lie outside the range of x
    if max(xi) > max(x) or min(xi) < min(x): # checking extremeties of xi vs extremeties of x
          return error
    
    yi = [] # initialize an empty list of output values (the values of the function at each point in xi)

    # calculate the function value at each point in xi
    for interp_point in xi: # interp_point represents a particular xi 
        # Find the x values that are just above and below interp_point
        x_upper = max(filter(lambda x: x <= interp_point, x)) # find the closest largest value of x (closest to interp_point)
        x_lower = min(filter(lambda x: x > interp_point, x)) # find the closest minimum value of x (closest to interp_point)

        # Get the corresponding values of y
        y_upper = fx[x.index(x_upper)]
        y_lower = fx[x.index(x_lower)]

        # calculate the slope of a linear interpolation line from x_lower to x_upper
        slope = (y_upper-y_lower)/(x_upper-x_lower) 
        dx = interp_point-x_lower # the difference between the interp_point and the lower x point
        
        interp_val = y_lower + slope*dx # interp_val-y_lower = slope(xi - x_lower), solve for interp_val
        yi+=[interp_val] # append the new interp_value to the list of yi outputs
    return yi

# Problem 2
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

    fxi = [] # return list that will include all the corresponding y values for xi.
    for interp in xi:
        x1 = max(filter(lambda x: x <= interp, x)) # upper limit for xi
        x2 = min(filter(lambda x: x > interp, x)) # lower limit for xi

        # Get the corresponding values of y
        y1 = fx[x.index(x1)]
        y2 = fx[x.index(x2)]
        i= np.where(np.isclose(x, x1)) # array of index of lower x point
        index = int(i[0][0]) # i is an array with one list (of length 1) as its value.

        # spline number i+1 
        val = a[index] +  r[3*index] * (interp-x1) + r[3*index+1] * (interp-x1)**2 + r[3*index+2] * (interp-x1)**3
        fxi+=[val] # append fx(xi) to list fxi
    return fxi

# Problem 3
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
    if np.sign(f(a))==np.sign(f(b)): 
        # intermediate value theorem does not apply
        return error
    
    num_it = 0
    m_old = a # arbitrary initialization; we know this to be an extreme case
    # assume a to be the left boundary and b to be the right boundary
    error = tol + 1 # arbitrary initialization for error to be greater than tolerance
    while error > tol and num_it < maxit: # 
        m = 1/2*(a+b) # midpoint, initial guess of root
        if np.sign(f(m))==np.sign(f(a)): 
            a = m # move left boundary closer
        else: # np.sign(f(m))==np.sign(f(b))
            b = m # move right boundary closer
        # calculate error on all iterations (excepting the first iteration)
        if num_it==0: # first iteration, no previous value of m to calculate relative error with
            pass
        else: # only calculate error after one iteration
            error = abs((m-m_old)/m) # absolute relative error (kinda) 
        m_old = m # update midpoint 
        num_it +=1 # update number of iterations being taken
    # return the approximation of the root, 
    # the value of the function at the root (~0),
    # the error associated with the calculated root, 
    # and the number of iterations taken to evaluate the root
    return m, f(m), error, num_it 

# Problem 4
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

    num_it = 0 # initialize number of iterations taken in method
    error = tol + 1 # arbitrary initialization for error to be greater than tolerance 
    # continue iterations while error exceeds specified tolerance and the number of iterations are within the max limit
    while error > tol and num_it < maxit: 
        # approximation for slope/f'(x0)
        fderiv = (f(x0 + delta*x0) - f(x0))/ (delta*x0)
        # fderiv*(x_new-x0) = f(x_new) - f(x0); assume f(x_new) = 0 (desired condition), and find x_new
        x_new = x0 - f(x0)/fderiv
        error = abs((x_new-x0)/x_new) # absolute relative error, kinda
        x0 = x_new # update new root estimate
        num_it +=1 # update number of iterations to solve
    # return the approximation of the root, 
    # the value of the function at the root (~0),
    # the error associated with the calculated root, 
    # and the number of iterations taken to evaluate the root
    return x_new, f(x_new), error, num_it 

###########################
# Problem 1 Test

x = [5, 2, 3, 20, 45, 1]
fx = [10, 4, 6, 40, 90, 2]
xi = [3, 7, 15, 1.5]
print(my_linear_interpolation(x,fx,xi))



# Problem 2 Test
'''
# x = [1, 2, 3, 4]
# fx = [1, 8, 27, 64]
# xi = [1.2, 2.4, 2.5]    

# x = [0,1,2]
# fx = [1,3,2]
# xi = [.5,1.5]

x1 = 0
x2 = 2*np.pi
ndata = 5
#data we have access to
x = np.linspace(x1,x2,ndata)
fx = np.sin(x)

#interpolate at a single point
xi = [(x1+x2)*.63]

print("calculated value", my_cubic_spline_interpolation(x,fx,xi))

from scipy.interpolate import CubicSpline
import scipy.interpolate
cs = CubicSpline(x, fx,bc_type='natural')
# print(x,f

f_cubic = CubicSpline(x, fx, bc_type='natural')

y_hat_cubic = f_cubic(xi)
# print(x,y)
print("a",y_hat_cubic)
print("interp val, splined scipy",cs(xi))
print("actual", np.sin(xi))


'''

# Problem 3 Test
'''
def f(x):
    return x**2 - 4 + x**3
a = -1
b = 3
tol = 1e-5
maxit = 100

# root, fx, ea, nit = my_bisection_method(f, a, b, tol, maxit)
print(my_bisection_method(f, a, b, tol, maxit))
###
def f(x):
    return np.sqrt(9.81*x/.25)*np.tanh(4*np.sqrt(9.81*.25/x)) - 36
a = 40
b = 200
tol = 1e-5
maxit = 100

root, fx, ea, nit = my_bisection_method(f, a, b, tol, maxit)
print(my_bisection_method(f, a, b, tol, maxit))

from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
x = np.linspace(40, 200, 100)
y = f(x)
plt.plot(x,y)
plt.plot(root,fx, "ro")
plt.grid()
plt.show()
plt.savefig("ex")
'''

# Problem 4 Test
'''

def f(x):
    return x**2-1
print(modified_secant_method(f,.1,.01, 100, .00001))
'''