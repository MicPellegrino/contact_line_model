"""
Visualizing the energy landscape in case of
1. Atomistically-flat surfaces;
2. Surfaces with strong/macroscopic defects;
3. Surfaces with weak/nanoscopic defects.
"""

import numpy as np
import matplotlib.pyplot as plt

cos = lambda t : np.cos(np.deg2rad(t))
sin = lambda t : np.sin(np.deg2rad(t))
sin2 = lambda t : sin(t)*sin(t)

theta0 = 90
theta = np.linspace(theta0-30,theta0+30,501)

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)

F_flat = (theta-theta0)*(cos(theta0)-cos(theta))
ax1.plot(theta,F_flat,'k--',linewidth=2)
ax2.plot(theta,F_flat,'k--',linewidth=2)
ax3.plot(theta,F_flat,'k--',linewidth=2)

eps = 0.1
a = 500/(2*np.pi)
F_rough = (theta-theta0)*(cos(theta0)-cos(theta)) + eps*sin2(a*theta)
ax1.plot(theta,F_rough,'k-',linewidth=3)
ax1.tick_params(axis='both',labelbottom=False,labelleft=False)

eps = 10
a = 150/(2*np.pi)
F_rough = (theta-theta0)*(cos(theta0)-cos(theta)) + eps*sin2(a*theta)
ax2.plot(theta,F_rough,'k-',linewidth=3)
ax2.tick_params(axis='both',labelbottom=False,labelleft=False)
ax2.set_ylabel('Wetting energy landscape',fontsize=30,labelpad=20)

eps = 2
a = 250/(2*np.pi)
F_rough = (theta-theta0)*(cos(theta0)-cos(theta)) + eps*sin2(a*theta)
ax3.plot(theta,F_rough,'k-',linewidth=3)
ax3.tick_params(axis='both',labelbottom=False,labelleft=False)
ax3.set_xlabel(r'$\theta_0$',fontsize=30,labelpad=20)

plt.show()