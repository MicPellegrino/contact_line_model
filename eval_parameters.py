import numpy as np
import scipy as sc
import scipy.special
import matplotlib.pyplot as plt
from math import isnan

# Cosine from angle in degrees
cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cos_m1 = lambda t : 0.0 if t>1.0 else np.rad2deg(np.arccos(t))
tan_m1 = lambda t : np.rad2deg(np.arctan(t))

# PARAMETERS

# NB this is the LOCAL equilibrium c.a. (i.e. Young)
theta_e = 70.0

k = 360.0

a = 3.0
print("a = "+str(a))

# Dummy value
# theta_g = 40.0
r = lambda a2 : (2.0/np.pi) * np.sqrt(a2+1.0) * sc.special.ellipe(a2/(a2+1.0))
theta_g = cos_m1(r(a**2)*cos(theta_e))
print("theta_W = "+str(theta_g))

delta_theta = theta_g-theta_e

print("delta_theta = "+str(delta_theta))

x_stat = (1.0/k)*cos_m1(-tan(delta_theta)/a)
x_stat = [x_stat, 1.0-x_stat]
print("Stationary points:")
print(x_stat)

x_0 = (1.0/k)*cos_m1(-tan(theta_g)/a)
x_0 = [x_0, 1.0-x_0]
print("Limit points:")
print(x_0)

# How much do the contact line jumps (w.r.t. x)?
if isnan(x_0[1]) :
    L_star = x_stat[1]-x_stat[0]
elif isnan(x_stat[1]) :
    L_star = 0.0
else :
    L_star = x_stat[1]-x_0[1]
print("lambda_star = "+str(L_star))

x = np.linspace(0.0, 1.0, 500)

plt.plot(x, cos(k*x), 'k-', linewidth=2.0)
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(delta_theta)/a, 'b-', linewidth=2.0, label=r'$\phi=\theta_0$')
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(theta_g)/a, 'r-', linewidth=2.0, label=r'$\phi=0$')
plt.xlim([0, 1.0])
plt.legend(fontsize=20.0)
plt.xlabel('x', fontsize=20.0)
plt.ylabel('cos(kx)', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()
