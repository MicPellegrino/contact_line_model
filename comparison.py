import numpy as np
import numpy.random as rng

# Possibily to remove after refactoring
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Needed for fitting the contact line friction coefficient
import scipy.optimize as sc_opt
import scipy.special as sc_spc

# To precompile some simple functions
# (the rest needs to be parallelized...)
from numba import jit

# Profiling (see "__main__")
import cProfile

# To speedup the SDE simulation
from mpi4py import MPI
MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()
MPI_ROOT = 0

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
# Bulk viscosity [mPa*s]
mu = 0.69
# Self expl.
TWOPI = 360
# Temperature [K]
T = 300
# Boltzmann constant [bar*nm^3/K]
kB = 0.1380649
# Depth [nm]
Ly = 4.67650

# Reduced droplet area [nondim]
@jit(nopython=True)
def fun_theta(t) :
    tt = np.deg2rad(t)
    return tt/(np.sin(tt)**2)-1.0/np.tan(tt)

# Helper to fsolve for the macroscopic angle
@jit(nopython=True)
def aux_fsolve(x,t) :
    return (x**2)*fun_theta(t)-np.pi


class RoughSubstrate :

    def __init__(self, l, mu_f, R0, a, theta_g_0_flat, theta_e, Gamma=None) :

        # Corrugation number [1/nm]
        self.k = TWOPI/l
        # Droplet radius [nm]
        self.R0 = R0
        # Roughness coefficient 'a' [nondim]
        self.a = a
        # Equilibrium c.a. on a flat surface [deg]
        self.theta_e = theta_e
        # Reference time [ns]
        self.tau = TWOPI*mu_f/(gamma*self.k)
        # Corrugation height [nm]
        h = a/self.k

        if MPI_RANK == MPI_ROOT :
            print("C.l. friction         = "+str(mu_f)+" [cP]")
            print("Corrugation length    = "+str(l)+" [nm]")
            print("Corrugation number    = "+str(self.k)+" [1/nm]")
            print("Droplet radius        = "+str(R0)+" [nm]")
            print("Roughness coefficient = "+str(a)+" [1]")
            print("Initial c.a. on flat  = "+str(theta_g_0_flat)+" [deg]")
            print("Equilibrium c. a.     = "+str(theta_e)+" [deg]")
            print("Reference time        = "+str(self.tau)+" [ns]")
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

        # Initial reduced droplet area [nondim]
        # self.fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )

        # Initial wetted distance [nm]
        self.x0 = R0*np.sqrt( np.pi/fun_theta(self.theta_g_0) )

        # Macroscopic angle, given by circular cap (input: coordinate rescaled over R0)
        self.theta_try = self.theta_g_0
        self.theta_g = lambda x : sc_opt.fsolve(lambda t : aux_fsolve(x,t), self.theta_try)[0]
        
        if Gamma == None :
            self.Gamma = (2*kB*T)/(Ly*l*10*gamma)
        else :
            self.Gamma = Gamma

        if MPI_RANK == MPI_ROOT :
            print("Initial c. a          = "+str(self.theta_g_0)+" [deg]")
            print("Initial c.l. distance = "+str(self.x0)+" [nm]")
            print('[TEST] : theta_g_0    = '+str(self.theta_g(self.x0/R0))+" [deg]")
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
        self.T_fin = t_fin/RS.tau
        self.T_bin = t_bin/RS.tau
        self.dt = 0.1*self.T_bin
        self.Nt = int(self.T_fin/self.dt)
        self.M = M

        if MPI_RANK == MPI_ROOT :
            print("Final time            = "+str(t_fin)+" [ns]")
            print("T_fin (nondim.)       = "+str(self.T_fin)+" [1]")
            print("dt (nondim.)          = "+str(self.dt)+" [1]")
            print("#replicas             = "+str(M))

    def simulate_ode(self, RS) :
        self.x_vec = []
        self.theta_g_vec = []
        x = RS.k*RS.x0/TWOPI
        ##### TO OPTIMIZE #####
        for n in range(self.Nt) :
            x = x + RS.V(x)*self.dt
            self.x_vec.append(x)
            RS.theta_try = RS.theta_g(RS.m2m*x)
            self.theta_g_vec.append(RS.theta_try)
        ##### ----------- #####
        self.x_vec = np.array(self.x_vec)
        self.theta_g_vec = np.array(self.theta_g_vec)
        self.t_vec = np.linspace(0.0, (self.Nt-1)*self.dt, self.Nt)

    def simulate_sde(self, RS) :
        self.x_ens = np.zeros(self.Nt)
        self.x2_ens = np.zeros(self.Nt)
        self.theta_g_ens = np.zeros(self.Nt)
        self.theta_g2_ens = np.zeros(self.Nt)
        x_vec_m = np.zeros(self.Nt)
        x_vec_m2 = np.zeros(self.Nt)
        theta_g_vec_m = np.zeros(self.Nt)
        theta_g_vec_m2 = np.zeros(self.Nt)
        ##### This could be trivially parallelized with mpi4py! #####
        # for m in range(self.M) :
        if MPI_RANK == MPI_ROOT :
            print("Simulating "+str(self.M)+" replicates")
        for m in range(MPI_RANK, self.M, MPI_SIZE) :
            # print("Rank "+str(MPI_RANK)+" is simulating replica "+str(m+1))
            x_vec_m_loc = []
            theta_g_vec_m_loc = []
            x = RS.k*RS.x0/TWOPI
            RS.theta_try = RS.theta_g_0
            rand_vec = rng.normal(0.0,1.0,self.Nt)
            ##### TO OPTIMIZE #####
            for n in range(self.Nt) :
                x = x + ( RS.V(x)*self.dt + np.sqrt(RS.Gamma*self.dt)*rand_vec[n]/RS.dsdx(x) )
                x_vec_m_loc.append(x)
                RS.theta_try = RS.theta_g(RS.m2m*x)
                theta_g_vec_m_loc.append(RS.theta_try)
            ##### ----------- #####
            x_vec_m_loc = np.array(x_vec_m_loc)
            x_vec_m += x_vec_m_loc
            x_vec_m2 += x_vec_m_loc*x_vec_m_loc
            theta_g_vec_m_loc = np.array(theta_g_vec_m_loc)
            theta_g_vec_m += theta_g_vec_m_loc
            theta_g_vec_m2 += theta_g_vec_m_loc*theta_g_vec_m_loc
            ##### Only these should be gathered/reduced by rank 0 #####
            # self.x_ens += x_vec_m
            # self.x2_ens += x_vec_m*x_vec_m
            # self.theta_g_ens += theta_g_vec_m
            # self.theta_g2_ens += theta_g_vec_m*theta_g_vec_m
            ##### ----------------------------------------------- #####
        MPI_COMM.Allreduce([x_vec_m, MPI.DOUBLE], [self.x_ens, MPI.DOUBLE], op=MPI.SUM)
        MPI_COMM.Allreduce([x_vec_m2, MPI.DOUBLE], [self.x2_ens, MPI.DOUBLE], op=MPI.SUM)
        MPI_COMM.Allreduce([theta_g_vec_m, MPI.DOUBLE], [self.theta_g_ens, MPI.DOUBLE], op=MPI.SUM)
        MPI_COMM.Allreduce([theta_g_vec_m2, MPI.DOUBLE], [self.theta_g2_ens, MPI.DOUBLE], op=MPI.SUM)
        ##### ------------------------------------------------- #####
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
        if MPI_RANK == MPI_ROOT :
            print("Eff. c.l. friction    = "+str(self.mu_f_fit)+" [cP]")

        return popt2
    
    def fit_cl_friction_ls(self, RS, lvel=10, lacc=1, p0_cf=None, p0_ls=[1,1,1,1,1], mv=1000) :

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
        popt2, _ = sc_opt.curve_fit(sinh, self.ct, self.v_fit, p0_cf, maxfev=mv)
        self.v_mkt = sinh(self.ct, *popt2)
        self.mu_f_fit = gamma/popt2[0]*popt2[2]
        if MPI_RANK == MPI_ROOT :
            print("Eff. c.l. friction    = "+str(self.mu_f_fit)+" [cP]")

        return popt2


#################################################################################################################
def testPlot(M=1) :

    """
        Checking how the solution looks like for different sets of parameters
    """

    # Flat surface
    # RS = RoughSubstrate(l=1,mu_f=10*mu,R0=20,a=0,theta_g_0_flat=105.8,theta_e=55.6)

    # 'Nice' rough surfaces
    RS = RoughSubstrate(l=1,mu_f=5.66,R0=20,a=0.2,theta_g_0_flat=105.8,theta_e=55.6)
    # RS = RoughSubstrate(l=1,mu_f=10*mu,R0=20,a=1,theta_g_0_flat=105.8,theta_e=55.6)
    # RS = RoughSubstrate(l=3.0357142857142856,mu_f=5.34,R0=15,a=1,theta_g_0_flat=105.8,theta_e=55.6)

    # 'Problematic' rough surfaces
    # RS = RoughSubstrate(l=4.388888888888889,mu_f=10*mu,R0=20,a=0.7777777777777777,theta_g_0_flat=105.8,theta_e=55.6)
    # RS = RoughSubstrate(l=3.5,mu_f=5.66,R0=15,a=1,theta_g_0_flat=105.8,theta_e=55.6)

    # One replicate
    # EM = EulerMurayama(RS=RS,t_fin=500,t_bin=0.1,M=M)

    # Several replicates
    EM = EulerMurayama(RS=RS,t_fin=30.0,t_bin=0.1,M=M)
    
    EM.simulate_ode(RS)
    EM.simulate_sde(RS)

    theta_w = RS.theta_w
    theta_fin_ode = EM.theta_g_vec[-1]
    theta_fin_sde = np.mean(EM.theta_g_ens[int(0.8*EM.Nt):])

    # Standard deviation (for fitting noise)
    x_lang_eq = TWOPI*EM.x_ens[int(0.5*EM.Nt):]/RS.k
    sigma = np.std(x_lang_eq)

    if MPI_RANK == MPI_ROOT :
        print("theta_w       = "+str(theta_w)+" [deg]")
        print("theta_fin_ode = "+str(theta_fin_ode)+" [deg]")
        print("theta_fin_sde = "+str(theta_fin_sde)+" [deg]")
        print("Standard deviation = ", sigma, " nm")

    if MPI_RANK == MPI_ROOT :
        fig1, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(RS.tau*EM.t_vec, TWOPI*EM.x_vec/RS.k, 'k-', linewidth=3.0, label=r'eq. 4')
        ax1.plot(RS.tau*EM.t_vec, TWOPI*EM.x_ens/RS.k, 'r-', linewidth=2.5, label=r'eq. 6')
        ax1.fill_between(RS.tau*EM.t_vec,TWOPI*(EM.x_ens+EM.x_std)/RS.k,TWOPI*(EM.x_ens-EM.x_std)/RS.k,color='r',alpha=0.5,linewidth=0.0)
        ax1.set_ylabel(r'$x_{cl}$ [nm]', fontsize=30.0)
        ax1.set_xlim([RS.tau*EM.t_vec[0], RS.tau*EM.t_vec[-1]])
        ax1.legend(fontsize=25)
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y', labelsize=25)
        ax2.plot(RS.tau*EM.t_vec, EM.theta_g_vec, 'k-', linewidth=3.0)
        ax2.plot(RS.tau*EM.t_vec, EM.theta_g_ens, 'r-', linewidth=2.0)
        ax2.fill_between(RS.tau*EM.t_vec,EM.theta_g_ens+EM.theta_std,EM.theta_g_ens-EM.theta_std,color='r',alpha=0.5,linewidth=0.0)
        ax2.plot(RS.tau*EM.t_vec, RS.theta_w*np.ones(EM.t_vec.shape), 'b--', linewidth=3, label=r'$\theta_W$')
        ax2.set_xlabel(r'$t$ [ns]', fontsize=30.0)
        ax2.set_ylabel(r'$\theta_g$ [deg]', fontsize=30.0)
        ax2.set_xlim([RS.tau*EM.t_vec[0], RS.tau*EM.t_vec[-1]])
        ax2.legend(fontsize=25)
        ax2.tick_params(axis='x', labelsize=25)
        ax2.tick_params(axis='y', labelsize=25)
        plt.show()

    p0 = None
    # EM.fit_cl_friction(RS,mv=1000)
    EM.fit_cl_friction_ls(RS,p0_cf=p0,mv=10000)

    if MPI_RANK == MPI_ROOT :
        
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(EM.t, EM.x_fit)
        ax1.plot(EM.t, EM.x)
        ax2.plot(EM.t, EM.v_fit)
        plt.show()

        plt.plot(EM.ct, EM.v_fit, 'k-', linewidth=3.0)
        plt.plot(EM.ct, EM.v_mkt, 'r-', linewidth=3.0, label='fit eq. 7')
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        plt.legend(fontsize=25)
        plt.xlabel(r'$\cos<\theta_g>$ []', fontsize=30.0)
        plt.ylabel(r'$<u_{cl}>$ [nm/ns]', fontsize=30.0)
        plt.show()


#################################################################################################################
def profile(M=1) :

    """
        Example run to profile the code
    """

    RS = RoughSubstrate(l=3.0357142857142856,mu_f=5.34,R0=15,a=1,theta_g_0_flat=105.8,theta_e=55.6)
    EM = EulerMurayama(RS=RS,t_fin=35.0,t_bin=0.1,M=M)
    EM.simulate_ode(RS)
    EM.simulate_sde(RS)
    EM.fit_cl_friction(RS)


#################################################################################################################
def optimize_noise(std_target,noise_ub,cl_friction=10,noise_lb=0,t_erg=1000,tol_std=0.01,tol_noise=0.001,maxit=100,plot=False) :

    """
        Attempt to "optimize" the noise given the standard deviation from MD
        It's stupid: you cannot use a simple bisection for this!!!
    """

    # Init sigma to some number that is larger than std_target+tol
    sigma = std_target*(1+10*tol_std)
    # Init noise in the middle
    noise = 0.5*(noise_ub+noise_lb)
    noise_pre = noise*(1+10*tol_noise)
    # Init optimization iterations
    it = 0
    
    while( np.abs(sigma-std_target)/std_target>tol_std and np.abs(noise-noise_pre)/noise_pre>tol_noise and it<maxit ) :
        # Initialize flat surface
        RS = RoughSubstrate(l=1,mu_f=cl_friction,R0=20,a=0,theta_g_0_flat=105.8,theta_e=55.6,Gamma=noise)
        # One replicate
        EM = EulerMurayama(RS=RS,t_fin=t_erg,t_bin=0.1,M=1)
        EM.simulate_sde(RS)
        # Standard deviation
        x_lang_eq = TWOPI*EM.x_ens[int(0.25*EM.Nt):]/RS.k
        x_lang_eq = x_lang_eq-np.mean(x_lang_eq)
        sigma = np.std(x_lang_eq)
        if sigma>std_target :
            noise_ub = noise 
            noise_lb = noise_lb
        else :
            noise_ub = noise_ub 
            noise_lb = noise
        noise_pre = noise
        noise = 0.5*(noise_ub+noise_lb)
        it += 1
        if MPI_RANK == MPI_ROOT :
            print("----- -------------------------------------- -----")
            print("iteration               ", it)
            print("std(c.l. position)    = ", sigma, " [nm]")
            print("----- -------------------------------------- -----")
    if MPI_RANK == MPI_ROOT :
        print("----- Convergence!                           -----")
        print("----- Noise (nondim.)     = ", noise, " [1]")
        print("----- -------------------------------------- -----")
    EM.simulate_ode(RS)

    if plot :

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

        nbins = 75
        counts, bins, bars = plt.hist(x_lang_eq,bins=nbins)
        plt.show()

    return noise

if __name__ == "__main__" :
    
    import io
    import pstats

    # REFERENCE VALUES
    # cl_friction_md = 5.659896689453016
    # noise_opt = 0.25
    if MPI_RANK == MPI_ROOT :
        print("### Baseline parameters ------------ ###")
        print("Surface tension      = "+str(gamma)+" [mPa*m]")
        print("Bulk viscosity       = "+str(mu)+" [mPa*s]")
        print("Temperature          = "+str(T)+" [K]")
        print("Box depth            = "+str(Ly)+" [nm]")

    # TESTING A SIMPLE SIMULATIONS
    testPlot(M=56)

    # PROFILING (ChatGPT code, to be cleaned...)
    """
    pr = cProfile.Profile()
    pr.enable()
    profile(M=56)
    pr.disable()
    if MPI_RANK == MPI_ROOT:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
    """
    
    # PRODUCTION RUNS (TO BE REFACTORED!)
    # noise_opt = optimize_noise(std_target=0.294,cl_friction=cl_friction_md,noise_ub=0.031)
