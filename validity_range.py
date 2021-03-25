import numpy as np
import matplotlib.pyplot as plt

l_vec = np.linspace(0.0, 1.0, 250)
h_cur = 5*l_vec**2

l_a = 0.2
l_R = 0.7
l_lims = np.linspace(l_a, l_R, 250)
l_curv = np.linspace(1.3/5, l_R, 250)

a1 = 0.8
a2 = 1.3

plt.plot(l_vec, a1*l_vec, 'k:', linewidth=2.0)
plt.plot(l_lims, a1*l_lims, 'g-', linewidth=4.5)
plt.plot(l_vec, a2*l_vec, 'k:', linewidth=2.0)
plt.plot(l_curv, a2*l_curv, 'g-', linewidth=4.5)
plt.plot(l_vec, h_cur, 'r-', linewidth=4.5)
plt.plot(l_a*np.array([1,1]), np.array([0,1]), 'r--', linewidth=4.5)
plt.plot(l_R*np.array([1,1]), np.array([0,1]), 'r-.', linewidth=4.5)
plt.xlabel(r'$\lambda$', fontsize=30.0)
plt.ylabel(r'$h$', fontsize=30.0)
plt.xlim([0,1])
plt.ylim([0,1])

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks along the bottom edge are off
    right=False,       # ticks along the top edge are off
    labelleft=False)   # labels along the bottom edge are off

plt.text(l_R-0.01, 0.01, r'$\lambda_R$',
        verticalalignment='bottom', horizontalalignment='right',
        color='red', fontsize=25)
plt.text(l_a-0.01, 0.01, r'$\lambda_d$',
        verticalalignment='bottom', horizontalalignment='right',
        color='red', fontsize=25)
plt.text(0.41, 0.7, r'$\kappa_0$',
        verticalalignment='bottom', horizontalalignment='right',
        color='red', fontsize=25)
plt.text(0.5, 0.45, r'$a=$const.',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=25)

plt.show()
