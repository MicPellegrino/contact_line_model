import numpy as np
import numpy.random as rng

import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.optimize as sc_opt
import scipy.special as sc_spc

# Rational polynomial fit
def rational(x, p, q) :
    return np.polyval(p, x) / np.polyval(q + [1.0], x)
def rational_4_2(x, p0, p1, p2, p3, q1) :
    return rational(x, [p0, p1, p2, p3], [q1,])
def ratioder_4_2(x, p0, p1, p2, p3, q1) :
    return rational(x, [2*p0*q1, 3*p0+p1*q1, 2*p1, p2-p3*q1], [q1*q1, 2*q1,])
def ratiodr2_4_2(x, p0, p1, p2, p3, q1) :
    return rational(x, [2*p0*q1*q1, 6*p0*q1, 6*p0, 2*(p1-p2*q1+p3*q1*q1)], [q1**3, 3*q1*q1, 3*q1])

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

# Hyperbolic sine for fitting c.l. friction
sinh = lambda x, a, b, c : a*np.sinh(b-c*x)

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

        # C.l. friction [cP]
        print("C.l. friction         = "+str(mu_f)+" [cP]")
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


# The numerical integrators need to be jitted!
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

    def fit_cl_friction(self, RS, p0=None, mv=1000) :

        self.t = RS.tau*self.t_vec
        self.x = TWOPI*self.x_ens/RS.k

        popt1, pcov1 = sc_opt.curve_fit(rational_4_2, self.t, self.x, maxfev=mv)
        self.x_fit = rational_4_2(self.t, *popt1)
        self.v_fit = ratioder_4_2(self.t, *popt1)

        self.ct = cos(self.theta_g_ens)
        popt2, pcov2 = sc_opt.curve_fit(sinh, self.ct, self.v_fit, p0, maxfev=mv)
        self.v_mkt = sinh(self.ct, *popt2)
        self.mu_f_fit = gamma/popt2[0]*popt2[2]
        print("Eff. c.l. friction    = "+str(self.mu_f_fit)+" [cP]")

        return popt2
    
    def fit_cl_friction_ls(self, RS, lvel=10, lacc=1, p0=None, p0_ls=[1,1,1,1,1], mv=1000) :

        self.t = RS.tau*self.t_vec
        self.x = TWOPI*self.x_ens/RS.k

        def rd_m(p) :
            r = ratioder_4_2(self.t,*p)
            r[r>0] = 0
            return r 
        
        def r2_p(p) :
            r = ratiodr2_4_2(self.t,*p)
            r[r<0] = 0
            return r 
        
        def fres(p):
            return np.concatenate((self.x-rational_4_2(self.t,*p),lvel*rd_m(p),lacc*r2_p(p)),axis=None)

        ls_results = sc_opt.least_squares(fres, x0=p0_ls, max_nfev=mv)
        popt1 = ls_results.x
        self.x_fit = rational_4_2(self.t, *popt1)
        self.v_fit = ratioder_4_2(self.t, *popt1)

        self.ct = cos(self.theta_g_ens)
        popt2, pcov2 = sc_opt.curve_fit(sinh, self.ct, self.v_fit, p0, maxfev=mv)
        self.v_mkt = sinh(self.ct, *popt2)
        self.mu_f_fit = gamma/popt2[0]*popt2[2]
        print("Eff. c.l. friction    = "+str(self.mu_f_fit)+" [cP]")

        return popt2


def test_plot() :

    # RS = RoughSubstrate(l=1,mu_f=10*mu,R0=20,a=1,theta_g_0_flat=105.8,theta_e=55.6)
    # RS = RoughSubstrate(l=4.388888888888889,mu_f=10*mu,R0=20,a=0.7777777777777777,theta_g_0_flat=105.8,theta_e=55.6)
    RS = RoughSubstrate(l=3.0357142857142856,mu_f=5.34,R0=15,a=1,theta_g_0_flat=105.8,theta_e=55.6)

    EM = EulerMurayama(RS=RS,t_fin=35.0,t_bin=0.1,M=20)
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

    EM.fit_cl_friction(RS)

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(EM.t, EM.x_fit)
    ax1.plot(EM.t, EM.x)
    ax2.plot(EM.t, EM.v_fit)
    plt.show()

    plt.plot(EM.ct, EM.v_fit, 'k-', linewidth=3.0)
    plt.plot(EM.ct, EM.v_mkt, 'r-', linewidth=3.0, label='MKT fit')
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    plt.legend(fontsize=25)
    plt.xlabel(r'$\cos<\theta_g>$ []', fontsize=30.0)
    plt.ylabel(r'$<u_{cl}>$ [nm/ns]', fontsize=30.0)
    plt.show()


def parametric_study(l_vec,a_vec,mu_f=10*mu,R0=20,theta_g_0_flat=105.8,theta_e=55.6,t_fin=100.0,t_bin=0.5,M=25) :

    s = (len(l_vec),len(a_vec))
    theta_w_vec = np.zeros(s)
    theta_fin_ode_vec = np.zeros(s)
    theta_fin_sde_vec = np.zeros(s)
    mu_f_ratio = np.zeros(s)

    p0 = None
    n = 0
    for i in range(len(l_vec)) :
        for j in range(len(a_vec)) :
            n += 1
            print("[ PROGRESS "+str(n)+"/"+str(theta_w_vec.size)+" ]")
            RS = RoughSubstrate(l=l_vec[i],mu_f=mu_f,R0=R0,a=a_vec[j],theta_g_0_flat=theta_g_0_flat,theta_e=theta_e)
            EM = EulerMurayama(RS=RS,t_fin=t_fin,t_bin=t_bin,M=M)
            EM.simulate_ode(RS)
            EM.simulate_sde(RS)
            p0 = EM.fit_cl_friction(RS,p0)
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
            mu_f_ratio[i,j] = EM.mu_f_fit/mu_f

    diff_ode = np.abs(theta_fin_ode_vec-theta_w_vec)
    diff_sde = np.abs(theta_fin_sde_vec-theta_w_vec)

    np.save('diff_ode.npy',diff_ode)
    np.save('diff_sde.npy',diff_sde)
    np.save('mu_f_ratio.npy',mu_f_ratio)


FSL = 50
FST = 35
LBP = 60

if __name__ == "__main__" :
    
    test_plot()

    Np = 36
    
    l_vec = np.linspace(0.25,3.5,Np)
    # print(l_vec)
    
    a_vec = np.linspace(0,1,Np)
    # print(a_vec)
    
    np.save('l_vec.npy',l_vec)
    np.save('a_vec.npy',a_vec)

    l_vec = np.load('l_vec.npy')
    a_vec = np.load('a_vec.npy')

    L, A = np.meshgrid(l_vec,a_vec,sparse=False,indexing='ij')

    # parametric_study(l_vec,a_vec,mu_f=5.34,R0=15,M=25,t_fin=100,t_bin=0.3)

    d1 = np.load('diff_ode.npy')
    d2 = np.load('diff_sde.npy')
    mr = np.load('mu_f_ratio.npy')

    vmax = max(np.max(d1),np.max(d2))
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    dmap1 = ax1.pcolormesh(L,A,d1,vmin=0,vmax=vmax,cmap=cm.plasma)
    ax1.set_xlabel('l [nm]',fontsize=FSL)
    ax1.set_ylabel('a [1]',fontsize=FSL)
    ax1.tick_params(labelsize=FST)
    cb1 = plt.colorbar(dmap1,ax=ax1)
    cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
    cb1.ax.tick_params(labelsize=0.8*FST)
    dmap2 = ax2.pcolormesh(L,A,d2,vmin=0,vmax=vmax,cmap=cm.plasma)
    ax2.set_xlabel('l [nm]',fontsize=FSL)
    ax2.set_ylabel('a [1]',fontsize=FSL)
    ax2.tick_params(labelsize=FST)
    cb2 = plt.colorbar(dmap2,ax=ax2)
    cb2.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
    cb2.ax.tick_params(labelsize=0.8*FST)
    plt.show()

    vmax = max(np.max(d1),np.max(d2))
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    dmap1 = ax1.pcolormesh(L,A,d2,vmin=0,vmax=vmax,cmap=cm.plasma)
    ax1.set_xlabel('l [nm]',fontsize=FSL)
    ax1.set_ylabel('a [1]',fontsize=FSL)
    ax1.tick_params(labelsize=FST)
    cb1 = plt.colorbar(dmap1,ax=ax1)
    cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
    cb1.ax.tick_params(labelsize=0.8*FST)
    dmap2 = ax2.pcolormesh(L,A,np.log(mr),cmap=cm.plasma)
    ax2.set_xlabel('l [nm]',fontsize=FSL)
    ax2.set_ylabel('a [1]',fontsize=FSL)
    ax2.tick_params(labelsize=FST)
    cb2 = plt.colorbar(dmap2,ax=ax2)
    cb2.ax.set_ylabel(r'$\log(\mu_f^*/\mu_f)$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
    cb2.ax.tick_params(labelsize=0.8*FST)
    plt.show()