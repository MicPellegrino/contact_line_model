import numpy as np

import matplotlib.pyplot as plt

# parameters

omega = 1.0
a = 2.0
theta_g = 20.0
theta_e = 19.0

# Cosine from angle in degrees
cos = lambda t : np.cos( np.deg2rad(t) )

# Microscopic angle
phi = lambda x : theta_g + np.arctan(a*cos(omega*x))

# Curvilinear coordinates measure
dsdx = lambda x : np.sqrt( 1.0 + a*a*(cos(omega*x)**2) )

# Potential
# MKT
V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / dsdx(x)
# PF
# V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / ( dsdx(x) * phi(x))

x = np.linspace(0.0, 360.0, 500)

plt.plot(x, V(x), 'k-', linewidth=1.5)
plt.plot([0.0, 360.0], [0.0, 0.0], 'r-', linewidth=1.5)
plt.xlim([0, 360.0])
plt.show()
