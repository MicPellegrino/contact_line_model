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
# theta_e = 110.0
k = 360.0

theta_e = 60
theta_g = 75
delta_theta = theta_g-theta_e

a_vec = np.linspace(0.25, 2.50, 100)
lambda_vec = []

for a in a_vec :

    x_stat = (1.0/k)*cos_m1(-tan(delta_theta)/a)
    x_stat = [x_stat, 1.0-x_stat]

    x_0 = (1.0/k)*cos_m1(-tan(theta_g)/a)
    x_0 = [x_0, 1.0-x_0]

    # How much do the contact line jumps (w.r.t. x)?
    if isnan(x_0[1]) :
        L_star = x_stat[1]-x_stat[0]
    elif isnan(x_stat[1]) :
        L_star = 0.0
    else :
        L_star = x_stat[1]-x_0[1]

    lambda_vec.append(L_star)

lambda_vec = np.array(lambda_vec)


"""
for a in a_vec:

    # Dummy value
    r = lambda a2 : (2.0/np.pi) * np.sqrt(a2+1.0) * sc.special.ellipe(a2/(a2+1.0))
    theta_g = cos_m1(r(a**2)*cos(theta_e))

    delta_theta = theta_g-theta_e

    x_stat = (1.0/k)*cos_m1(-tan(delta_theta)/a)
    x_stat = [x_stat, 1.0-x_stat]

    x_0 = (1.0/k)*cos_m1(-tan(theta_g)/a)
    x_0 = [x_0, 1.0-x_0]

    # How much do the contact line jumps (w.r.t. x)?
    if isnan(x_0[1]) :
        L_star = x_stat[1]-x_stat[0]
    elif isnan(x_stat[1]) :
        L_star = 0.0
    else :
        L_star = x_stat[1]-x_0[1]

    lambda_vec.append(L_star)

lambda_vec = np.array(lambda_vec)
"""

plt.plot(a_vec, lambda_vec, 'k-')
plt.xlabel('a', fontsize=20.0)
plt.ylabel(r'$\lambda^*$', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()
