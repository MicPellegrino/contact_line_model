import numpy as np

import matplotlib.pyplot as plt

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

theta_g = 56.89947694
theta_e = 20.0
delta_theta = theta_g-theta_e

k = 360.0
a = 0.95*tan(delta_theta)
print("a = "+str(a))

# Microscopic angle
phi = lambda x : theta_g + tan_m1(a*cos(k*x))

# Curvilinear coordinates measure
dsdx = lambda x : np.sqrt( 1.0 + a*a*(cos(k*x)**2) )

# Prefactor function
f = lambda x : sin( phi(x) )

# Potential
# MKT
V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / ( dsdx(x)*f(x) )

"""
x = np.linspace(0.0, 1.0, 500)
plt.plot(x, V(x), 'k-', linewidth=1.5)
plt.plot([0.0, 1.0], [0.0, 0.0], 'r-', linewidth=1.5)
plt.xlim([0, 1.0])
plt.show()
"""

# Numerical integration
dt = 0.1
Nt = 10000
Np = 10.0

x0 = 0.0
x_vec = []
x = x0
n = 0
while n<Nt and x<Np :
    x = x + V(x)*dt
    x_vec.append(x)
    n+=1

x_vec = np.array(x_vec)
t_vec = np.linspace(0.0, n*dt, n)

plt.plot(t_vec, x_vec, 'k-', linewidth=3.0)
plt.xlabel('t [-1]', fontsize=30.0)
plt.ylabel('x [-1]', fontsize=30.0)
plt.xlim([t_vec[0], t_vec[-1]])
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Numerical integration', fontsize=30.0)
plt.show()
