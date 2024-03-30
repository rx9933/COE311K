from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('seaborn-poster')

x1 = 0
x2 = 2*np.pi
ndata = 5
#data we have access to
x = np.linspace(x1,x2,ndata)
y = np.sin(x)


#establish interpolating function
f = interp1d(x, y)
#interpolate at a single point
interp_point = (x1+x2)*.63
y_hat = f(interp_point)
print(y_hat)

#for plotting purposes only
x_true = np.linspace(x1,x2,100)
y_true = np.sin(x_true)
#for plotting purposes only
interp_points = np.linspace(x1,x2,100)
y_hats = f(interp_points)



plt.figure(figsize = (10,8))
plt.plot(x_true,y_true,label='true data')
plt.scatter(x, y,label='provided data')
plt.plot(interp_points, y_hats, '--',label='linear interpolant line')
plt.scatter(interp_point,y_hat,c='r',label='linear interpolated values')
plt.title('Linear vs Cubic Interpolation at x = '+str(interp_point))
plt.xlabel('x')
plt.ylabel('y')



#cubic spline
# use bc_type = 'natural' adds the constraints as we described above
f_cubic = CubicSpline(x, y, bc_type='natural')

y_hat_cubic = f_cubic(interp_point)
print("a",y_hat)


#for plotting purposes only
y_hats_cubic = f_cubic(interp_points)

plt.plot(interp_points, y_hats_cubic, '--',label='cubic interpolant line')
plt.scatter(interp_point,y_hat_cubic,c='b',label='cubic interpolated values')

plt.grid()
plt.legend()

plt.show()
# plt.close()

'''
#matrix from notes example
A = np.array([[1,1,1,0,0,0],[0,0,0,1,1,1],[1,2,3,-1,0,0],[0,2,6,0,-2,0],[0,1,0,0,0,0],[0,0,0,0,2,6]])
b = np.array([2,-1,0,0,0,0])
a = np.linalg.solve(A,b)

x = np.array([0.0,1.0,2.0])
fx = np.array([1.0,3.0,2.0])
xi = np.array([0.5,1.5])

#some points between 0 and 1
x_plot = np.linspace(0,1,100)
s1 = fx[0]+a[0]*(x_plot-x[0])+a[1]*(x_plot-x[0])**2+a[2]*(x_plot-x[0])**3
plt.plot(x_plot,s1,'--',label = '$s_1$(x)')
#some points between 1 and 2
x_plot2 = np.linspace(1,2,100)
s2 = fx[1]+a[3]*(x_plot2-x[1]) + a[4]*(x_plot2-x[1])**2 + a[5]*(x_plot2-x[1])**3
plt.plot(x_plot2,s2,'--',label='$s_2$(x)')
plt.scatter(x,fx,c='r',label='(x,fx)')
#mark interpolated points, one is defined by spline 1, the other by spline 2
fxi= np.array([fx[0]+a[0]*(xi[0]-x[0])+a[1]*(xi[0]-x[0])**2+a[2]*(xi[0]-x[0])**3,
               fx[1]+a[3]*(xi[1]-x[1]) + a[4]*(xi[1]-x[1])**2 + a[5]*(xi[1]-x[1])**3 ])
print(fxi)
plt.scatter(xi,fxi,c='orange',label='(xi,fxi)')
plt.title('Cubic Interpolation from notes')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()

plt.show()
'''