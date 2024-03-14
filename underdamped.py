import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

print("Simulating a underdamped Langevin equation with no external forcing")

# Nondimensional
tau = 1e-9              # [s]
Ly = 4.67e-9            # [m]
surf_tens = 5.78e-2     # [Pa*m]
viscosity = 8.77e-4     # [Pa*s]
kBT = 4.14e-21          # [J]

# Fixed parameters
x0 = 0.0
u0 = 0.0
muf = 3.2
mcl = 1e-13                # [kg/m]

# Numerical parameters
dt = 0.001
N = 10000

# Derived quantities
eta_1 = (kBT*viscosity)/(surf_tens*surf_tens*tau*Ly)
eta_2 = (kBT*viscosity*viscosity)/(Ly*mcl*surf_tens*surf_tens)
gamma = (muf*viscosity*tau)/mcl
alpha = 1-np.exp(-gamma*dt)
t_max = dt*N
print("eta_1 = ", eta_1)
print("eta_2 = ", eta_2)
print("gamma = ", gamma)
print("alpha = ", alpha)
print("t_max = ", t_max)

# Initialize arrays
x = np.zeros(N)
x[0] = x0
u = np.zeros(N)
u[0] = u0
rg = rng.normal(size=N)
t = dt*np.linspace(0,N-1,N)
du = 0
utemp = 0

for k in range(N-1) :

    utemp = u[k]

    # du = -alpha*utemp + np.sqrt( kBT*(1-alpha**2)/mcl ) * rg[k+1]
    du = -alpha*utemp + np.sqrt( eta_2*alpha*(2-alpha) ) * rg[k+1]
    # du = -alpha*utemp + np.sqrt( eta_2*(1-alpha**2) ) * rg[k+1]

    x[k+1] = x[k] + dt*(utemp+0.5*du)

    u[k+1] = utemp + du

# Check: run overdamped Langevin equation
x_over = np.zeros(N)
x_over[0] = x0

for k in range(N-1) :

    x_over[k+1] = x_over[k] + np.sqrt(2*eta_1*dt/muf) * rg[k+1]

plt.plot(t, x_over, label='over')
plt.plot(t, x, label='under')
plt.legend()
plt.show()
