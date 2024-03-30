import numpy as np
def f(x):
    # replace with any function to find optimization of 
    return x*np.exp(-x**2)# xe^(-x^2)

def df(x, error):
    # use taylor series centered expansion for f'(x)
    h = error/10000
    return (f(x+h)-f(x-h))/(2*h)

def d2f(x, error):
    h = error/10000    
    # use taylor series centered expansion for f''(x)
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2
    
def find_root(f, df, x_prev, maxit, tol):
    num_it = 0
    error = tol + 1 # arbitrary initialization for error to be greater than tol
    while error > tol and num_it < maxit:
        # x_new = x - f(x) / df(x) # Newton-Raphson update, when finding root of f(x)
        x_new = x_prev - df(x_prev, error) / d2f(x_prev, error) # Newton-Raphson update, when finding optimization of f(x) (root of f'(x))
        error = abs((x_new - x_prev)/x_prev)
        x_prev = x_new # update previous iteration, xn 
        num_it += 1 # increment iteration count
    return x_new, num_it, error

# calculate value
root, iterations, error = find_root(f, df, 1, 10, 1e-6)
# output value
print(f"The optimization is at x = {root}. Result found in {iterations} iterations with error {error}.")
