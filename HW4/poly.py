import numpy as np
def f_deriv(x,delta=.00001):
    """
    Approximates the derivative of a function at a given point using the finite difference method.

    Parameters:
    - f: The function to differentiate.
    - x: The point at which to evaluate the derivative.
    - h: The small value used for approximation (default is 1e-5).

    Returns:
    - The approximate derivative of f at x.
    """
    # return (f(x + h) - f(x - h)) / (2 * h)
    # return (f(x) - f(x_prev))/(x-x_prev)
    return (f(x+delta)-f(x))/(delta)

def f(x):
    return x*np.exp(-x**2)
    # orig = 1/4*x**4 - x
    # return x**3 -1
def find_root(maxit, tol, x):
    num_it = 0
    error = tol + 1 # arbitrary initialization for error to be greater than togut commit -m "tangent/second methods"
    x_old = x
    while error > tol and num_it < maxit: # 
        # x_new = x_old - f_deriv(x_old)/f_deriv(f_deriv(x_old))
        x_new = x_old - x_old/f_deriv(x_old)
        error = abs((x_new-x_old)/x_new)
        num_it +=1
        x_old = x_new
    return x_new, num_it, error
       
print(find_root(54, 10**-2,.7)) 
                # iter, tol, x

