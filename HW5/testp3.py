from matplotlib import pyplot as plt

x= np.linspace(-10,10, 100)
fx = np.cos(x)
xi = np.linspace(-10,10, 100)
h=1
plt.plot(x,fx,label = "true")
plt.plot(x,fourth_order_diff(f,xi,h),label = "approx")
plt.show()
plt.savefig("p3")


def f(x):
    return np.sin(x)
