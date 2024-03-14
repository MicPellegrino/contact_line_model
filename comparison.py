import numpy as np
import numpy.random as rng

import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.optimize as sc_opt
import scipy.special as sc_spc

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

# GLOBAL VARIABLES (FOR WATER-VAPOR)
# Surface tension [mPa*m]
gamma = 57.8
print("Surface tension       = "+str(gamma)+" [mPa*m]")
# Bulk viscosity [mPa*s]
mu = 0.69
print("Bulk viscosity        = "+str(mu)+" [mPa*s]")
# Self expl.
TWOPI = 360
# Temperature [K]
T = 300
# Boltzmann constant [bar*nm^3/K]
kB = 0.1380649
# Depth [nm]
Ly = 4.67650


class RoughSubstrate :

    def __init__(self, l, mu_f, R0, a, theta_g_0_flat, theta_e, Gamma=None) :

        # Corrugation wavelength [nm]
        print("Corrugation length    = "+str(l)+" [nm]")
        # Corrugation number [1/nm]
        self.k = TWOPI/l
        print("Corrugation number    = "+str(self.k)+" [1/nm]")
        # Droplet radius [nm]
        self.R0 = R0
        print("Droplet radius        = "+str(R0)+" [nm]")
        # Roughness coefficient 'a' [nondim]
        self.a = a
        print("Roughness coefficient = "+str(a)+" [1]")
        # Initial c.a. on a flat surface [deg]
        print("Initial c.a. on flat  = "+str(theta_g_0_flat)+" [deg]")
        # Equilibrium c.a. on a flat surface [deg]
        self.theta_e = theta_e
        print("Equilibrium c. a.     = "+str(theta_e)+" [deg]")
        # Reference time [ns]
        self.tau = TWOPI*mu_f/(gamma*self.k)
        print("Reference time        = "+str(self.tau)+" [ns]")
        # Corrugation height [nm]
        h = a/self.k
        print("Corrugation height    = "+str(h)+" [nm]")

        ### Conversion from 'micro-to-macro' ###
        self.m2m = TWOPI/(self.k*R0)
        ### Wenzel law ###
        a2 = a**2
        self.rough_parameter = (2.0/np.pi) * np.sqrt(a2+1.0) * sc_spc.ellipe(a2/(a2+1.0))
        ### Eq. Wenzel angle ###
        self.theta_w = cos_m1(self.rough_parameter*cos(theta_e))
        # Contact angles [deg]
        self.theta_g_0 = cos_m1(self.rough_parameter*cos(theta_g_0_flat))
        print("Initial c. a          = "+str(self.theta_g_0)+" [deg]")

        # Initial reduced droplet area [nondim]
        self.fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )
        # Initial wetted distance [nm]
        self.x0 = R0*np.sqrt( np.pi/self.fun_theta(self.theta_g_0) )
        print("Initial c.l. distance = "+str(self.x0)+" [nm]")

        # Macroscopic angle, given by circular cap (input: coordinate rescaled over R0)
        self.theta_try = self.theta_g_0
        self.theta_g = lambda x : sc_opt.fsolve(lambda t : ( (x**2)*self.fun_theta(t)-np.pi ), self.theta_try)[0]
        print('[TEST] : theta_g_0    = '+str(self.theta_g(self.x0/R0))+" [deg]")
        
        if Gamma == None :
            self.Gamma = (2*kB*T)/(Ly*l*10*gamma)
        else :
            self.Gamma = Gamma
        print("Noise (nondim.)       = "+str(self.Gamma)+" [1]")

        # Microscopic angle (input: coordinate rescaled over the wavelength)
        self.phi = lambda x : self.theta_g(self.m2m*x) + tan_m1(self.a*cos(TWOPI*x))
        # Curvilinear coordinates measure (input: coordinate rescaled over the wavelength)
        self.dsdx = lambda x : np.sqrt(1.0+(self.a*cos(TWOPI*x))**2)
        # MKT Potential (input: coordinate rescaled over the wavelength)
        self.V = lambda x : (cos(self.theta_e)-cos(self.phi(x)))/self.dsdx(x)


class EulerMurayama :

    def __init__(self, RS, t_fin, t_bin, M) :
        print("Final time            = "+str(t_fin)+" [ns]")
        self.T_fin = t_fin/RS.tau
        self.T_bin = t_bin/RS.tau
        self.dt = 0.1*self.T_bin
        self.Nt = int(self.T_fin/self.dt)
        print("T_fin (nondim.)       = "+str(self.T_fin)+" [1]")
        print("dt (nondim.)          = "+str(self.dt)+" [1]")
        self.M = M
        print("#replicas             = "+str(M))

    def simulate_ode(self, RS) :
        self.x_vec = []
        self.theta_g_vec = []
        x = RS.k*RS.x0/TWOPI
        for n in range(self.Nt) :
            x = x + RS.V(x)*self.dt
            self.x_vec.append(x)
            RS.theta_try = RS.theta_g(RS.m2m*x)
            self.theta_g_vec.append(RS.theta_try)
        self.x_vec = np.array(self.x_vec)
        self.t_vec = np.linspace(0.0, (self.Nt-1)*self.dt, self.Nt)

    def simulate_sde(self, RS) :
        self.x_ens = np.zeros(self.Nt)
        self.x2_ens = np.zeros(self.Nt)
        self.theta_g_ens = np.zeros(self.Nt)
        self.theta_g2_ens = np.zeros(self.Nt)
        for m in range(self.M) :
            print("Sim. replica "+str(m+1))
            x_vec_m = []
            theta_g_vec_m = []
            x = RS.k*RS.x0/TWOPI
            RS.theta_try = RS.theta_g_0
            for n in range(self.Nt) :
                x = x + ( RS.V(x)*self.dt + np.sqrt(RS.Gamma*self.dt)*rng.normal(0.0,1.0)/RS.dsdx(x) )
                x_vec_m.append(x)
                RS.theta_try = RS.theta_g(RS.m2m*x)
                theta_g_vec_m.append(RS.theta_try)
            x_vec_m = np.array(x_vec_m)
            theta_g_vec_m = np.array(theta_g_vec_m)
            self.x_ens += x_vec_m
            self.x2_ens += x_vec_m*x_vec_m
            self.theta_g_ens += theta_g_vec_m
            self.theta_g2_ens += theta_g_vec_m*theta_g_vec_m
        self.x2_ens /= self.M
        self.x_ens /= self.M
        self.x_var = (self.x2_ens-self.x_ens*self.x_ens)
        self.x_std = np.sqrt(self.x_var)
        self.x_ste = self.x_std/np.sqrt(self.M)
        self.theta_g2_ens /= self.M
        self.theta_g_ens /= self.M
        self.theta_var = (self.theta_g2_ens-self.theta_g_ens*self.theta_g_ens)
        self.theta_std = np.sqrt(self.theta_var)
        self.theta_ste = self.theta_std/np.sqrt(self.M)


def test_plot() :

    RS = RoughSubstrate(l=1,mu_f=10*mu,R0=20,a=0.1,theta_g_0_flat=105.8,theta_e=55.6)

    EM = EulerMurayama(RS=RS,t_fin=100.0,t_bin=0.5,M=25)
    EM.simulate_ode(RS)
    EM.simulate_sde(RS)

    theta_w = RS.theta_w
    print("theta_w       = "+str(theta_w)+" [deg]")
    theta_fin_ode = EM.theta_g_vec[-1]
    print("theta_fin_ode = "+str(theta_fin_ode)+" [deg]")
    theta_fin_sde = np.mean(EM.theta_g_ens[int(0.8*EM.Nt):])
    print("theta_fin_sde = "+str(theta_fin_sde)+" [deg]")

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(RS.tau*EM.t_vec, TWOPI*EM.x_vec/RS.k, 'k-', linewidth=3.0)
    ax1.plot(RS.tau*EM.t_vec, TWOPI*EM.x_ens/RS.k, 'r-', linewidth=2.5)
    ax1.fill_between(RS.tau*EM.t_vec,TWOPI*(EM.x_ens+EM.x_std)/RS.k,TWOPI*(EM.x_ens-EM.x_std)/RS.k,color='r',alpha=0.5,linewidth=0.0)
    ax1.set_ylabel(r'$x_{cl}$ [nm]', fontsize=30.0)
    ax1.set_xlim([RS.tau*EM.t_vec[0], RS.tau*EM.t_vec[-1]])
    ax1.tick_params(axis='x',which='both',labelbottom=False)
    ax1.tick_params(axis='y', labelsize=25)

    ax2.plot(RS.tau*EM.t_vec, EM.theta_g_vec, 'k-', linewidth=3.0)
    ax2.plot(RS.tau*EM.t_vec, EM.theta_g_ens, 'r-', linewidth=2.0, label=r'$<x_{cl}>$')
    ax2.fill_between(RS.tau*EM.t_vec,EM.theta_g_ens+EM.theta_std,EM.theta_g_ens-EM.theta_std,color='r',alpha=0.5,linewidth=0.0)
    ax2.plot(RS.tau*EM.t_vec, RS.theta_w*np.ones(EM.t_vec.shape), 'b--', linewidth=3, label=r'$\theta_W$')
    ax2.set_xlabel(r'$t$ [ns]', fontsize=30.0)
    ax2.set_ylabel(r'$\theta_g$ [deg]', fontsize=30.0)
    ax2.set_xlim([RS.tau*EM.t_vec[0], RS.tau*EM.t_vec[-1]])
    ax2.legend(fontsize=25)
    ax2.tick_params(axis='x', labelsize=25)
    ax2.tick_params(axis='y', labelsize=25)

    plt.show()


def parametric_study(l_vec,a_vec,mu_f=10*mu,R0=20,theta_g_0_flat=105.8,theta_e=55.6,t_fin=100.0,t_bin=0.5,M=25) :

    s = (len(l_vec),len(a_vec))
    theta_w_vec = np.zeros(s)
    theta_fin_ode_vec = np.zeros(s)
    theta_fin_sde_vec = np.zeros(s)

    n = 0
    for i in range(len(l_vec)) :
        for j in range(len(a_vec)) :
            n += 1
            print("[ PROGRESS "+str(n)+"/"+str(theta_w_vec.size)+" ]")
            RS = RoughSubstrate(l=l_vec[i],mu_f=mu_f,R0=R0,a=a_vec[j],theta_g_0_flat=theta_g_0_flat,theta_e=theta_e)
            EM = EulerMurayama(RS=RS,t_fin=t_fin,t_bin=t_bin,M=M)
            EM.simulate_ode(RS)
            EM.simulate_sde(RS)
            theta_w = RS.theta_w
            print("theta_w       = "+str(theta_w)+" [deg]")
            theta_fin_ode = EM.theta_g_vec[-1]
            print("theta_fin_ode = "+str(theta_fin_ode)+" [deg]")
            theta_fin_sde = np.mean(EM.theta_g_ens[int(0.8*EM.Nt):])
            print("theta_fin_sde = "+str(theta_fin_sde)+" [deg]")
            print("# ######################################## #")
            theta_w_vec[i,j] = theta_w
            theta_fin_ode_vec[i,j] = theta_fin_ode
            theta_fin_sde_vec[i,j] = theta_fin_sde

    diff_ode = np.abs(theta_fin_ode_vec-theta_w_vec)
    diff_sde = np.abs(theta_fin_sde_vec-theta_w_vec)

    np.save('diff_ode.npy',diff_ode)
    np.save('diff_sde.npy',diff_sde)

if __name__ == "__main__" :
    
    Np = 36
    l_vec = np.linspace(0.5,5.5,Np)
    print(l_vec)
    a_vec = np.linspace(0,1,Np)
    print(a_vec)
    np.save('l_vec.npy',l_vec)
    np.save('a_vec.npy',a_vec)

    # Testing
    l_vec = np.load('l_vec.npy')
    a_vec = np.load('a_vec.npy')

    L, A = np.meshgrid(l_vec,a_vec,sparse=False,indexing='ij')

    # parametric_study(l_vec,a_vec,M=25)

    d1 = np.load('diff_ode.npy')
    d2 = np.load('diff_sde.npy')
    vmax = max(np.max(d1),np.max(d2))
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    dmap1 = ax1.pcolormesh(L,A,d1,vmin=0,vmax=vmax,cmap=cm.bone)
    ax1.set_xlabel('l [nm]')
    ax1.set_ylabel('a [1]')
    cb1 = plt.colorbar(dmap1,ax=ax1)
    cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270)
    dmap2 = ax2.pcolormesh(L,A,d2,vmin=0,vmax=vmax,cmap=cm.bone)
    ax2.set_xlabel('l [nm]')
    ax2.set_ylabel('a [1]')
    cb2 = plt.colorbar(dmap2,ax=ax2)
    cb2.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270)
    plt.show()

    plt.pcolormesh(L,A,d1-d2,cmap=cm.bone)
    plt.colorbar()
    plt.show()