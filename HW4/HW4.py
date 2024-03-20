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
    if xi > max(x) or xi < min(x):
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
print(my_linear_interpolation(x,fx,7))

def my_cubic_spline_interpolation(x,fx,xi):
    '''
    Takes in x - input data
    fx - data you want to fit curve to, should be same length as x
    xi - the points you wish to interpolate at

    return fxi, a vector same size as xi with interpolated values using cubic splines

    return error if any points in xi lie outside the range of x
    '''
    if xi > max(x) or xi < min(x):
        return error
    ai = fx


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
