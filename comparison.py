import numpy as np
import numpy.random as rng

import matplotlib.pyplot as plt

import scipy.optimize as sc_opt
import scipy.special as sc_spc

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

theta_g_0 = 130.0
theta_e = 38.8
delta_theta = theta_g_0-theta_e
r0 = 15     # Units of periods

fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )
A0 = ( 0.5*r0**2 ) * fun_theta( theta_g_0 )

k = 360.0
"""
curvature_coeff = 0.05
a = curvature_coeff*tan(delta_theta)
a2 = a**2
"""
a = 0.7
a2 = a**2
print("a = "+str(a))

# Wenzel relation
rough_parameter = (2.0/np.pi) * np.sqrt(a2+1.0) * sc_spc.ellipe(a2/(a2+1.0))
theta_w = cos_m1( rough_parameter*cos(theta_e) )
print("theta_w = "+str(theta_w))

# Langevin coefficient (constant)
Gamma = 0.001

# Macroscopic angle (given by circular cap)
fun_area = lambda x : 2*A0 / ( (r0+x)**2 )
theta_try = theta_g_0
theta_g = lambda x : sc_opt.fsolve(lambda t : (fun_theta(t)-fun_area(x)), theta_try)[0]

# Microscopic angle
phi = lambda x : theta_g(x) + tan_m1(a*cos(k*x))

# Curvilinear coordinates measure
dsdx = lambda x : np.sqrt( 1.0 + a*a*(cos(k*x)**2) )

# Prefactor function
# f = lambda x : sin( phi(x) )
f = lambda x : 1.0

# Potential
# MKT
V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / ( dsdx(x)*f(x) )

# Numerical integration: deterministic
N_fin = 750
dt = 0.1
Nt = int(N_fin/dt)
x0 = 0.0
x_vec = []
theta_g_vec = []
x = x0
n = 0
while n<Nt :
    x = x + V(x)*dt
    x_vec.append(x)
    theta_g_vec.append(theta_g(x))
    theta_try = theta_g(x)
    n+=1

x_vec = np.array(x_vec)
t_vec = np.linspace(0.0, n*dt, n)

# Stochastic replicas
n = 0
M = 1
x_ens = np.zeros(Nt)
theta_g_ens = np.zeros(Nt)
# Replicas
for m in range(M) :
    print("Sim. replica "+str(m+1))
    x_vec_m = []
    theta_g_vec_m = []
    x = x0
    for n in range(Nt) :
        x = x + V(x)*dt + np.sqrt(Gamma*dt)*rng.normal(0.0, 1.0)/(dsdx(x)*f(x))
        x_vec_m.append(x)
        theta_g_vec_m.append(theta_g(x))
    x_vec_m = np.array(x_vec_m)
    theta_g_vec_m = np.array(theta_g_vec_m)
    x_ens += x_vec_m
    theta_g_ens += theta_g_vec_m
x_ens /= M
theta_g_ens /= M

plt.plot(t_vec, x_ens+r0, 'r-', linewidth=2.0, label='stochastic model, '+str(M)+' replicas')
plt.plot(t_vec, x_vec+r0, 'k--', linewidth=3.0, label='toy model')
plt.xlabel('t [-1]', fontsize=30.0)
plt.ylabel('x [-1]', fontsize=30.0)
plt.xlim([t_vec[0], t_vec[-1]])
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Numerical integration, position', fontsize=30.0)
plt.legend(fontsize=20.0, loc='lower right')
plt.show()

plt.plot(t_vec, theta_g_ens, 'r-', linewidth=2.0, label='stochastic model, '+str(M)+' replicas')
plt.plot(t_vec, theta_g_vec, 'k--', linewidth=3.0, label='toy model')
plt.plot(t_vec, theta_w*np.ones(t_vec.shape), 'b-', linewidth=2.5, label='Wenzel')
plt.xlabel('t [-1]', fontsize=30.0)
plt.ylabel(r'$\theta_g$ [deg]', fontsize=30.0)
plt.xlim([t_vec[0], t_vec[-1]])
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Numerical integration, global angle', fontsize=30.0)
plt.legend(fontsize=20.0, loc='upper right')
plt.show()
