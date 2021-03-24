"""
    DISCLAIMER
    The Langevin model implemented here is really crude and does NOT respect
    fluctuation-dissipation theroem in general (or does it?).
    It serves only for coding purposes.
"""

import numpy as np
import numpy.random as rng

import matplotlib.pyplot as plt

import scipy.optimize as sc_opt

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

theta_g_0 = 85.0
# theta_g_0 = 56.89947694
theta_e = 20.0
delta_theta = theta_g_0-theta_e

# Cylindrical droplet; init. parameters
r0 = 10     # Units of periods
# fun_theta = lambda t : ( np.deg2rad(t) - sin(t) ) / ( sin(t)**2 )
fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )
A0 = ( 0.5*r0**2 ) * fun_theta( theta_g_0 )

k = 360.0
curvature_coeff = 0.1
a = curvature_coeff*tan(delta_theta)
print("a = "+str(a))

# Langevin coefficient (constant)
Gamma = 0.1
# Gamma = 0.0

# Macroscopic angle (given by circular cap)
fun_area = lambda x : 2*A0 / ( (r0+x)**2 )
theta_g = lambda x : sc_opt.fsolve(lambda t : (fun_theta(t)-fun_area(x)), theta_e)[0]

# Microscopic angle
phi = lambda x : theta_g(x) + tan_m1(a*cos(k*x))

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
T = 100
Nt = int(T/dt)

x0 = 0.0
n = 0
M = 1

x_ens = np.zeros(Nt)

# Replicas
for m in range(M) :

    print("Sim. replica "+str(m))

    x_vec = []
    x = x0

    for n in range(Nt) :
        x = x + V(x)*dt + np.sqrt(Gamma*dt)*rng.normal(0.0, 1.0)
        x_vec.append(x)

    x_vec = np.array(x_vec)
    x_ens += x_vec

x_ens /= M

t_vec = np.linspace(0.0, Nt*dt, Nt)

plt.plot(t_vec, x_ens+r0, 'k-', linewidth=3.0)
plt.xlabel('t [-1]', fontsize=30.0)
plt.ylabel('x [-1]', fontsize=30.0)
plt.xlim([t_vec[0], t_vec[-1]])
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Numerical integration', fontsize=30.0)
plt.show()
