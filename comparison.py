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

### REDUCED UNITS ###

# Droplet radius [nm]
R0 = 25.0
print("Droplet radius         = "+str(R0)+" [nm]")
# Corrugation spacing [1/nm]
k = 360.0/4.0
print("Corrugation spacing    = "+str(k)+" [1/nm]")
# Displacement prefactor [nondim]
beta = R0*k
print("Displacement prefactor = "+str(beta)+" [nondim]")
# Bulk viscosity [mPa*s]
mu = 0.887
print("Bulk viscosity         = "+str(mu)+" [mPa*s]")
# Surface tension [mPa*m]
gamma = 57.8
print("Surface tension        = "+str(gamma)+" [mPa*m]")
# Reference speed [nm/ps]
U_ref = (gamma/mu)
print("Reference speed        = "+str(U_ref)+" [nm/ns]")
# Reference time [ps]
tau = R0/U_ref
print("Reference time         = "+str(tau)+" [ns]")

# Contact angles [deg]
theta_g_0 = 130.0
print("Initial c. a           = "+str(theta_g_0)+" [deg]")
theta_e = 38.8
print("Equilibrium c. a.      = "+str(theta_e)+" [deg]")
delta_theta = theta_g_0-theta_e

# Roughness coefficient 'a' [nondim]
a = 0.7
# a = 0.0
print("Roughness coefficient  = "+str(a)+" [nondim]")
# Corrugation height [nm]
h = a/k
print("Corrugation height     = "+str(h)+" [nm]")

# Wenzel
a2 = a**2
rough_parameter = (2.0/np.pi) * np.sqrt(a2+1.0) * sc_spc.ellipe(a2/(a2+1.0))
theta_w = cos_m1( rough_parameter*cos(theta_e) )

# Initial reduced droplet area [nondim]
A0 = np.pi
fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )
# Initial wetted distance [nondim]
x0 = np.sqrt( np.pi/fun_theta(theta_g_0) )
print("Initial c.l. distance  = "+str(x0*R0)+" [nm]")

# Friction ratio [nondim]
mu_star = 10.0
print("Friction ratio mu_f/mu = "+str(mu_star)+" [nm]")

# Langevin coefficient (constant)
Gamma = 0.00616
# Gamma = 0.01

# Macroscopic angle (given by circular cap)
theta_try = theta_g_0
theta_g = lambda x : sc_opt.fsolve(lambda t : ( (x**2)*fun_theta(t)-np.pi ), theta_try)[0]

# Microscopic angle
phi = lambda x : theta_g(x) + tan_m1(a*cos(beta*x))

# Curvilinear coordinates measure
dsdx = lambda x : np.sqrt( 1.0 + (a*cos(beta*x))**2 )

# Potential
# MKT
V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / dsdx(x)

# Numerical integration
# Final time [ps]
t_fin = 100.0
t_bin = 0.01
T_fin = t_fin/tau
T_bin = t_bin/tau
dt = 0.1*T_bin
Nt = int(T_fin/dt)

print("Time-step              = "+str(dt*tau)+" [ns]")

x_vec = []
theta_g_vec = []
x = x0

for n in range(Nt) :
    x = x + V(x)*dt/mu_star
    x_vec.append(x)
    theta_try = theta_g(x)
    theta_g_vec.append(theta_try)

x_vec = np.array(x_vec)
t_vec = np.linspace(0.0, n*dt, Nt)

# Stochastic replicas
M = 1
x_ens = np.zeros(Nt)
theta_g_ens = np.zeros(Nt)
# Replicas
for m in range(M) :
    print("Sim. replica "+str(m+1))
    x_vec_m = []
    theta_g_vec_m = []
    x = x0
    theta_try = theta_g_0
    for n in range(Nt) :
        x = x + ( V(x)*dt + np.sqrt(Gamma*dt)*rng.normal(0.0, 1.0)/dsdx(x) )/mu_star
        x_vec_m.append(x)
        theta_try = theta_g(x)
        theta_g_vec_m.append(theta_try)
    x_vec_m = np.array(x_vec_m)
    theta_g_vec_m = np.array(theta_g_vec_m)
    x_ens += x_vec_m
    theta_g_ens += theta_g_vec_m
x_ens /= M
theta_g_ens /= M

fig1, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(tau*t_vec, R0*x_vec, 'k-', linewidth=3.0)
ax1.plot(tau*t_vec, R0*x_ens, 'r-', linewidth=2.0, label='stochastic model, '+str(M)+' replicas')
ax1.set_xlabel('t [ns]', fontsize=30.0)
ax1.set_ylabel('x [ns]', fontsize=30.0)
ax1.set_xlim([tau*t_vec[0], tau*t_vec[-1]])
ax1.tick_params(axis='x', labelsize=25)
ax1.tick_params(axis='y', labelsize=25)
ax1.set_title('Contact line position', fontsize=30.0)

ax2.plot(tau*t_vec, theta_g_vec, 'k-', linewidth=3.0)
ax2.plot(tau*t_vec, theta_g_ens, 'r-', linewidth=2.0, label='stochastic model, '+str(M)+' replicas')
ax2.plot(tau*t_vec, theta_w*np.ones(t_vec.shape), 'k--', linewidth=2.5)
ax2.set_xlabel('t [ns]', fontsize=30.0)
ax2.set_ylabel(r'$\theta_g$ [deg]', fontsize=30.0)
ax2.set_xlim([tau*t_vec[0], tau*t_vec[-1]])
ax2.tick_params(axis='x', labelsize=25)
ax2.tick_params(axis='y', labelsize=25)
ax2.set_title('Global contact angle', fontsize=30.0)

plt.show()

"""
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
"""
