# Classes
from parametric_study import parametricStudy

# Global variables for MPI
from comparison import MPI_COMM, MPI_RANK, MPI_SIZE, MPI_ROOT

# Other import
import numpy as np

# To plot with 'latex-like' font
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def productionRun(FSL=25, FST=20, LBP=35, noise=None, cl_friction=10, R0_md=15, theta_e_md=55.6, t_fin_md=15,
                cah_plot_cutoff=None, clf_plot_cutoff=None, M=25, Np=40) :

    l_vec = np.linspace(0.25,3.5,Np)
    a_vec = np.linspace(0,1.0,Np)

    f_tag = '{0:.3f}'.format(cl_friction)
    n_tag = '{0:.3f}'.format(noise)

    L, A = np.meshgrid(l_vec,a_vec,sparse=False,indexing='ij')

    parametricStudy(noise,l_vec,a_vec,mu_f=cl_friction,R0=R0_md,theta_g_0_flat=101.2,theta_e=theta_e_md,M=M,t_fin=t_fin_md,t_bin=0.2)

    if MPI_RANK == MPI_ROOT :
        d1 = np.load('diff_ode.npy')
        d2 = np.load('diff_sde.npy')
        mr = np.load('mu_f_ratio.npy')

    if MPI_RANK == MPI_ROOT :

        if cah_plot_cutoff==None :
            cah_plot_cutoff = np.max(d2)

        if clf_plot_cutoff==None :
            clf_plot_cutoff = np.percentile(np.log(mr), 95)

        fig1, (ax1, ax2) = plt.subplots(1, 2)
        dmap1 = ax1.pcolormesh(L,A,d1,vmin=0,vmax=np.max(d1),cmap=cm.plasma)
        ax1.set_xlabel('l [nm]',fontsize=FSL)
        ax1.set_ylabel('a [1]',fontsize=FSL)
        ax1.tick_params(labelsize=FST)
        cb1 = plt.colorbar(dmap1,ax=ax1)
        cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb1.ax.tick_params(labelsize=0.8*FST)
        dmap2 = ax2.pcolormesh(L,A,d2,vmin=0,vmax=cah_plot_cutoff,cmap=cm.plasma)
        ax2.set_xlabel('l [nm]',fontsize=FSL)
        ax2.set_ylabel('a [1]',fontsize=FSL)
        ax2.tick_params(labelsize=FST)
        cb2 = plt.colorbar(dmap2,ax=ax2)
        cb2.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb2.ax.tick_params(labelsize=0.8*FST)
        plt.tight_layout()
        plt.savefig("figures/a-hysteresis_muf="+f_tag+"_ns="+n_tag+".png",format='png')

    if MPI_RANK == MPI_ROOT :

        fig1, (ax1, ax2) = plt.subplots(1, 2)
        dmap1 = ax1.pcolormesh(L,A,d2,vmin=0,vmax=cah_plot_cutoff,cmap=cm.plasma)
        ax1.set_xlabel('l [nm]',fontsize=FSL)
        ax1.set_ylabel('a [1]',fontsize=FSL)
        ax1.tick_params(labelsize=FST)
        cb1 = plt.colorbar(dmap1,ax=ax1)
        cb1.ax.set_ylabel(r'$|\theta_{\infty}-\theta_W|$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb1.ax.tick_params(labelsize=0.8*FST)
        dmap2 = ax2.pcolormesh(L,A,np.log(mr),vmin=1,vmax=clf_plot_cutoff,cmap=cm.plasma)
        ax2.set_xlabel('l [nm]',fontsize=FSL)
        ax2.set_ylabel('a [1]',fontsize=FSL)
        ax2.tick_params(labelsize=FST)
        cb2 = plt.colorbar(dmap2,ax=ax2)
        cb2.ax.set_ylabel(r'$\log(\mu_f^*/\mu_f)$', rotation=270,fontsize=0.8*FSL,labelpad=LBP)
        cb2.ax.tick_params(labelsize=0.8*FST)
        plt.tight_layout()
        plt.savefig("figures/friction-amplification_muf="+f_tag+"_ns="+n_tag+".png",format='png')


if __name__ == "__main__" :

    cl_friction_md = 5.659896689453016
    noise_opt = 0.25
    productionRun(noise=noise_opt,cl_friction=cl_friction_md,cah_plot_cutoff=10,clf_plot_cutoff=5,M=56)