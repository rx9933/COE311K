# Taylor series for e^x

from math import exp, factorial
import numpy as np
import matplotlib.pyplot as plt
def exp_approx(x, tolerence, maxit):
    rel_err = 9999
    fx = 0.0
    fxold = 0.0
    iter = 0
    while (rel_err > tolerence) and (iter < maxit):
        fx += x**iter / factorial(iter) # np.math.factorial(x)
        iter +=1
        rel_err = abs(fx - fxold)/fx
        fxold = fx
    return fx, rel_err, iter


x = 4.8954
etol = 1e-5
print(f"Actual e^{x}: {exp(x)}")

maxit = 100
fxApprox, rel_err, it_num = exp_approx(x,etol, maxit)
print(f"calculated: {fxApprox}\nrelative error: {rel_err}\niterations: {it_num}")

###################################################################################
# Plotting
x = np.linspace(0,10,100) # .1 between x
fx = x**2

plt.plot(x, fx, label = "f(x) = x**2")
plt.title("x vs f(x)")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.legend()
plt.grid()
plt.show()
plt.savefig("plot_exampl.png")