# Classes
from comparison import RoughSubstrate, EulerMurayama

# Global variables for MPI
from comparison import MPI_COMM, MPI_RANK, MPI_SIZE, MPI_ROOT
# Global physical variables
from comparison import gamma, mu, T, kB, Ly

# Other import
import numpy as np

def parametricStudy(noise,l_vec,a_vec,mu_f=10,R0=20,theta_g_0_flat=105.8,theta_e=55.6,t_fin=100.0,t_bin=0.5,M=25,mvfit=10000) :

    s = (len(l_vec),len(a_vec))
    theta_w_vec = np.zeros(s)
    theta_fin_ode_vec = np.zeros(s)
    theta_fin_sde_vec = np.zeros(s)
    mu_f_ratio = np.zeros(s)

    p0 = None
    n = 0
    for i in range(len(l_vec)) :
        for j in range(len(a_vec)) :
            noise = (2*kB*T)/(Ly*l_vec[i]*10*gamma)
            n += 1
            if MPI_RANK == MPI_ROOT :
                print("[ PROGRESS "+str(n)+"/"+str(theta_w_vec.size)+" ]")
            RS = RoughSubstrate(l=l_vec[i],mu_f=mu_f,R0=R0,a=a_vec[j],theta_g_0_flat=theta_g_0_flat,theta_e=theta_e,Gamma=noise)
            EM = EulerMurayama(RS=RS,t_fin=t_fin,t_bin=t_bin,M=M)
            EM.simulate_ode(RS)
            EM.simulate_sde(RS)
            # p0 = EM.fit_cl_friction(RS,p0,mv=mvfit)
            p0 = EM.fit_cl_friction_ls(RS,p0_cf=p0,mv=mvfit)
            theta_w = RS.theta_w
            theta_fin_ode = EM.theta_g_vec[-1]
            theta_fin_sde = np.mean(EM.theta_g_ens[int(0.8*EM.Nt):])
            theta_w_vec[i,j] = theta_w
            theta_fin_ode_vec[i,j] = theta_fin_ode
            theta_fin_sde_vec[i,j] = theta_fin_sde
            mu_f_ratio[i,j] = EM.mu_f_fit/mu_f
            if MPI_RANK == MPI_ROOT :
                print("theta_w       = "+str(theta_w)+" [deg]")
                print("theta_fin_ode = "+str(theta_fin_ode)+" [deg]")
                print("theta_fin_sde = "+str(theta_fin_sde)+" [deg]")
                print("# ######################################## #")

    diff_ode = np.abs(theta_fin_ode_vec-theta_w_vec)
    diff_sde = np.abs(theta_fin_sde_vec-theta_w_vec)

    if MPI_RANK == MPI_ROOT :
        np.save('diff_ode.npy',diff_ode)
        np.save('diff_sde.npy',diff_sde)
        np.save('mu_f_ratio.npy',mu_f_ratio)


if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    from comparison import mu

    # Thermal-capillary length
    l_th = 0.27

    Np = 40
    # Np = 20
    
    # noise = 0.25
    noise = None
    # noise = (2*kB*T)/(Ly*l*10*gamma)

    l_vec = np.linspace(0.20,2.0,Np)
    a_vec = np.linspace(0,1.0,Np)
    L, A = np.meshgrid(l_vec/l_th,a_vec,sparse=False,indexing='ij')

    parametricStudy(noise,l_vec,a_vec,mu_f=10*mu,R0=20,theta_g_0_flat=105.8,theta_e=55.6,t_fin=100.0,t_bin=0.5,M=56,mvfit=10000)

    if MPI_RANK == MPI_ROOT :

        FSL=25
        FST=20
        LBP=35

        d1 = np.load('diff_ode.npy')
        d2 = np.load('diff_sde.npy')
        mr = np.load('mu_f_ratio.npy')

        cah_plot_cutoff = np.max(d2)
        clf_plot_cutoff = np.percentile(np.log(mr), 95)

        # fig1, (ax1, ax2) = plt.subplots(2, 1)
        ax1 = plt.subplot(211)
        dmap1 = ax1.pcolormesh(L,A,d1,vmin=0,vmax=np.max(d1),cmap=cm.plasma)
        # ax1.set_xlabel('l [nm]',fontsize=FSL)
        ax1.set_ylabel(r'$a$ []',fontsize=FSL)
        ax1.tick_params(labelsize=FST)
        cb1 = plt.colorbar(dmap1,ax=ax1)
        cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb1.ax.tick_params(labelsize=0.8*FST)
        ax2 = plt.subplot(212,sharex=ax1)
        dmap2 = ax2.pcolormesh(L,A,d2,vmin=0,vmax=cah_plot_cutoff,cmap=cm.plasma)
        ax2.set_xlabel(r'$l/l_{th}$ []',fontsize=FSL)
        ax2.set_ylabel(r'$a$ []',fontsize=FSL)
        ax2.tick_params(labelsize=FST)
        cb2 = plt.colorbar(dmap2,ax=ax2)
        cb2.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb2.ax.tick_params(labelsize=0.8*FST)
        plt.tight_layout()
        plt.show()

        # fig1, (ax1, ax2) = plt.subplots(2, 1)
        ax1 = plt.subplot(211)
        dmap1 = ax1.pcolormesh(L,A,d2,vmin=0,vmax=cah_plot_cutoff,cmap=cm.plasma)
        # ax1.set_xlabel('l [nm]',fontsize=FSL)
        ax1.set_ylabel(r'$a$ []',fontsize=FSL)
        ax1.tick_params(labelsize=FST)
        cb1 = plt.colorbar(dmap1,ax=ax1)
        cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb1.ax.tick_params(labelsize=0.8*FST)
        ax2 = plt.subplot(212,sharex=ax1)
        dmap2 = ax2.pcolormesh(L,A,np.log(mr),vmin=1,vmax=clf_plot_cutoff,cmap=cm.plasma)
        ax2.set_xlabel(r'$l/l_{th}$ []',fontsize=FSL)
        ax2.set_ylabel(r'$a$ []',fontsize=FSL)
        ax2.tick_params(labelsize=FST)
        cb2 = plt.colorbar(dmap2,ax=ax2)
        cb2.ax.set_ylabel(r'$\log(\mu_f^*/\mu_f)$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb2.ax.tick_params(labelsize=0.8*FST)
        plt.tight_layout()
        plt.show()