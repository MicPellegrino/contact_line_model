import numpy as np

import matplotlib.pyplot as plt

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

theta_g = 35.0
theta_e = 20.0
delta_theta = theta_g-theta_e

k = 360.0
a_crit = tan(delta_theta)

a = np.array([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])*a_crit

# Prefactor function
f = lambda x : sin( phi(x) )

# Numerical integration
dt = 0.1
Nt = 500

# Initial conditions
x0 = 0.0

x_vec = dict()

for a_val in a :

    x_vec[a_val] = []

    # Microscopic angle
    phi = lambda x : theta_g + tan_m1(a_val*cos(k*x))
    # Curvilinear coordinates measure
    dsdx = lambda x : np.sqrt( 1.0 + a_val*a_val*(cos(k*x)**2) )
    # Potential
    V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / ( dsdx(x)*f(x) )

    x = x0
    for n in range(Nt) :
        x = x + V(x)*dt
        x_vec[a_val].append(x)

    x_vec[a_val] = np.array(x_vec[a_val])

t_vec = np.linspace(0.0, Nt*dt, Nt)

for a_val in a :
    plt.plot(t_vec, x_vec[a_val], label=str(a_val))
plt.legend()
plt.show()
