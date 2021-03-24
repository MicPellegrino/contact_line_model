import numpy as np

import matplotlib.pyplot as plt

import scipy.optimize as sc_opt

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

Lx = 175.0
d = 50.0

fun_theta = lambda t : ( np.deg2rad(t) - sin(t) ) / ( 1-cos(t) )
area_factor = np.pi*(d/Lx)**2
theta_0 = 10.0

theta_min = sc_opt.fsolve( lambda t : fun_theta(2.0*t) - area_factor, theta_0 )
print(theta_min)
