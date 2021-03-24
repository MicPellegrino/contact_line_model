import numpy as np

import matplotlib.pyplot as plt

# Cosine from angle in degrees
cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

# parameters

theta_g = 35.0
theta_e = 20.0
delta_theta = theta_g-theta_e

k = 360.0

a1 = tan(delta_theta)
a2 = 2.00*tan(delta_theta)
a3 = 0.75*tan(delta_theta)
print(a1)

x_0 = (1.0/k)*cos_m1(-tan(delta_theta)/a2)
x_0 = [x_0, 1.0-x_0]
print(x_0)

x = np.linspace(0.0, 1.0, 500)

h = a2/k
plt.plot(x, cos(k*x), 'k-', linewidth=2.0)
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(delta_theta)/a1, 'r-', linewidth=2.0, label="a="+str(a1))
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(delta_theta)/a2, 'b-', linewidth=2.0, label="a="+str(a2))
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(delta_theta)/a3, 'g-', linewidth=2.0, label="a="+str(a3))
plt.xlim([0, 1.0])
plt.legend(fontsize=20.0)
plt.xlabel('x', fontsize=20.0)
plt.ylabel('cos(kx)', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

"""
plt.plot(x, cos(k*x), 'k--', linewidth=1.5)
plt.plot(x, h*sin(k*x), 'b-', linewidth=2.0)
plt.plot([0.0, 1.0], -np.array([1.0, 1.0]) * \
    tan(delta_theta)/a2, 'r--', linewidth=1.5)
plt.plot(x_0, h*sin(k*np.array(x_0)), 'bo', linewidth=2.0, markersize=7.5)
plt.xlim([0, 1.0])
plt.show()
"""
