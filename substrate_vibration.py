import numpy as np
import matplotlib.pyplot as plt

"""
epsilon = 1.77      # kJ/mol
sigma   = 0.277021  # nm
"""
epsilon = 1.770     # kJ/mol
sigma   = 0.366     # nm

kappa   = 10000     # kJ/mol
res_pos = 0.366     # nm
temp    = 300       # K
k_b     = 0.823e10  # nm^2*amu*ps^-2*K^-1
mass    = 2*15.999  # amu

V_LJ    = lambda x : 4.0*epsilon*(sigma/x)**12
V_RES   = lambda x : 0.5*kappa*(x-res_pos)**2

F_LJ    = lambda x : (48.0*epsilon/x)*(sigma/x)**12
F_RES   = lambda x : -kappa*(x-res_pos)

x_0 = res_pos
# v_0 = np.sqrt(k_b*temp/(3.0*mass))
v_0 = 0.0

acc     = lambda x : ( F_LJ(x) + F_RES(x) ) / mass

T       = 10
dt      = 0.002
N       = int(T/dt)

x = np.zeros(N, dtype=float)
x[0] = x_0
v = np.zeros(N, dtype=float)
v[0] = v_0
t = np.zeros(N, dtype=float)
t[0] = 0.0

for n in range(N-1) :
    v[n+1] = v[n] + 0.5*acc(x[n])*dt
    x[n+1] = x[n] + v[n+1]*dt
    v[n+1] = v[n+1] + 0.5*acc(x[n+1])*dt
    t[n+1] = t[n]+dt

plt.plot(t, x-res_pos, 'b-')
# plt.plot(t, v, 'r-')
plt.show()
