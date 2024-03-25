import numpy as np
def f_deriv(x,delta):
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
    return (f(x+delta*x)-f(x))/(delta*x)

def f(x):
    # return x*np.exp(-x**2)
    # orig = 1/4*x**4 - x
    return x**3 -1
def find_root(maxit, tol, x):
    num_it = 0
    error = tol + 1 # arbitrary initialization for error to be greater than tolerance
    x_old = x # initialize to initial guess
    while error > tol and num_it < maxit: # 
        x_new = x_old - f(x_old)/f_deriv(x_old,.01)
        error = abs((x_new-x_old)/x_new)
        num_it +=1
        x_old = x_new
    return x_new, num_it, error
       
print(find_root(20, 10**-4, -.3))

# x = 2
# derivative = f_deriv(f, x)
# print("The derivative of f at x =", x, "is approximately", derivative)
