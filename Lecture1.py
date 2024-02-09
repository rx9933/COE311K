import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5,5,11)
y = 3*x**2 + 3
# plt.plot(x,y)
# plt.show()

# m=2
# print(m==4 or m==3)
f = lambda x: x+3
print(f(3))

def aux_func(x):
    x+=5
    return x
def aux_func_2(x):
    x+=6
    return x
def big_func(some_func, x):
    x = some_func(x)
    return x
y = 6
z = big_func(aux_func, y)
z2 = big_func(aux_func_2, y)
print(z)
print(z2)



