from scipy.interpolate import interp1d # write out interp1d
import matplotlib.pyplot as plt 
# plt.style.use('seaborn-poster')
import numpy as np
# x = [0,1,2]
x1 = 0
x2 = 2*np.pi
ndata = 200
# y = [1,3,2]
x = np.linspace(x1,x2,ndata)
y = np.sin(x)
# interp_point = 1.5
interp_point = (x1+x2)*.63


x_true = np.linspace(x1,x2,100)
y_true = np.sin(x_true)

f = interp1d(x,y)
y_hat = f(interp_point)
plt.figure(figsize = (10,8))
plt.plot(x_true, y_true)
plt.scatter(x,y)
plt.plot(interp_point, y_hat, 'ro')
plt.show()
print("interp val ", y_hat, "vs actual ", np.sin(interp_point))
print("relative error = ", (y_hat-np.sin(interp_point))/np.sin(interp_point))