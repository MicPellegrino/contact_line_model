import numpy as np
import matplotlib.pyplot as plt

cos = lambda t : np.cos(np.deg2rad(t))
sin = lambda t : np.sin(np.deg2rad(t))

theta_vec = np.linspace(50, 130, 500)

a = 1.0

ener_barr = (0.5*sin(theta_vec)+cos(theta_vec))**2

U_mkt = -cos(theta_vec)

U_jr = -cos(theta_vec)*np.exp(a*(0.25-ener_barr))

u_cr = 0.2

i_low_mkt = np.argmin(np.abs(U_mkt+u_cr))
i_upp_mkt = np.argmin(np.abs(U_mkt-u_cr))

i_low_jr = np.argmin(np.abs(U_jr+u_cr))
i_upp_jr = np.argmin(np.abs(U_jr-u_cr))

plt.plot([np.min(theta_vec-90), np.max(theta_vec-90)], [0, 0], 'k:', linewidth=1.75)
plt.plot([0, 0], [-0.8, 0.8], 'k:', linewidth=1.75)

plt.plot([np.min(theta_vec-90), theta_vec[i_low_mkt]-90], 
    [U_mkt[i_low_mkt], U_mkt[i_low_mkt]], 'k-', linewidth=1.75)
plt.plot([np.min(theta_vec-90), theta_vec[i_upp_mkt]-90], 
    [U_mkt[i_upp_mkt], U_mkt[i_upp_mkt]], 'k-', linewidth=1.75)

plt.plot(theta_vec-90, U_mkt, 'b-', linewidth=3.5, label='Eq. (1)')
plt.plot(theta_vec-90, U_jr, 'r-.', linewidth=3.5, label='Eq. (2)')

plt.fill_between(theta_vec[i_low_mkt:i_upp_mkt]-90, 
    U_mkt[i_low_mkt:i_upp_mkt], y2=-0.8, alpha=0.5)

plt.fill_between(theta_vec[i_low_jr:i_upp_jr]-90, 
    U_jr[i_low_jr:i_upp_jr], y2=-0.8, alpha=0.5)

plt.title('(b)', y=1.08, fontsize=45, loc='left', fontweight="bold")

plt.legend(fontsize=30)
plt.xlim([np.min(theta_vec-90), np.max(theta_vec-90)])
plt.ylim([-0.8, 0.8])
plt.xlabel(r'$\theta-\theta_0$ [deg]', fontsize=50)
plt.ylabel(r'Ca$_{cl}$ [1]', fontsize=50)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.show()
