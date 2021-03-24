from math import pi

import scipy as sc
import scipy.special
import scipy.optimize as sc_opt

import numpy as np
import matplotlib.pyplot as plt

# a2 = amplitude^2 * frequency^2
rough_parameter = lambda a2 : (2.0/pi) * np.sqrt(a2+1.0) * sc.special.ellipe(a2/(a2+1.0))

"""
q_O (e)     theta_0 (deg.) (approx)
-0.40       110
-0.60       90
-0.67       70
-0.74       37
-0.79       0
The case on complete wetting is not considered
"""

# Substrate partial charges and respective contact angles
charges = np.array([0.40, 0.60, 0.67, 0.74, 0.79])
flat_angles = np.array([110.0, 90.0, 70.0, 37.0, 0.0])

# Roughness parameter
a = np.zeros(8, dtype=float)
a[0] = 0.0
for k in range(1,8) :
    a[k] = 0.25+0.25*k
xi_r = rough_parameter(a**2)

# Storage for contact angles
cos_wave_angle = np.outer( xi_r, np.cos( np.deg2rad(flat_angles) ) )
sat_arccos = lambda cval : 0.0 if (cval>1.0) else np.rad2deg(np.arccos(cval))
wave_angle = np.vectorize(sat_arccos)(cos_wave_angle)

print("-------------------------------------------------------------------")
print("# Effect of surface roughness on equilibrium c.a. in Wenzel state #")
print("-------------------------------------------------------------------")

line = "q \ a\t"
for i in range(8) :
    line = line+str(a[i])+"\t"
print(line)

for i in range(5):
    line = str(charges[i])+"\t"
    for j in range (8):
        line = line+"{:3.3f}".format(wave_angle[j][i])+"\t"
    print(line)

plt.plot(a, wave_angle, 'o--')
plt.xlabel('a [nondim.]')
plt.ylabel('Contact angle [deg]')
plt.show()

# Obtaining the value of a for the correct c.a.
theta_Y_tar = 38.8  # [deg]
theta_W_tar = 30.0  # [deg]

r_tar = np.cos( np.deg2rad(theta_W_tar) ) / np.cos( np.deg2rad(theta_Y_tar) )
a_tar = sc_opt.fsolve(lambda a : rough_parameter(a**2)-r_tar, 0.1)[0]

print("------------------------------------")
print("theta_Y = "+str(theta_Y_tar)+" [deg]")
print("theta_W = "+str(theta_W_tar)+" [deg]")
print("r       = "+str(r_tar)+" [-1]")
print("a       = "+str(a_tar)+" [-1]")
