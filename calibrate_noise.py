import numpy as np
import scipy.special as sc_spc

from comparison import RoughSubstrate, EulerMurayama
from comparison import TWOPI, MPI_COMM, MPI_RANK, MPI_SIZE, MPI_ROOT

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

AX_LABELS_FS = 17.5
AX_TICKS_FS = 12.5
LEGEND_FS = 15

# Some useful definitions...
cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

# Load MD simulation output
n_tags = ['06','08','10','12','14','16']
n_range = [6,8,10,12,14,16]
a_tags = ['02','04','06','08','10']
a_range = [0.2,0.4,0.6,0.8,1.0]
md_output = dict()
for nt in n_tags :
    for at in a_tags :
        md_output[nt+at] = np.load("MD_DATA/data_a"+at+"_n"+nt+".npz")

# Number of replicates for the Langevin model
m_rep = 56
# Time-step for the EM method
t_bin_em = 0.1

# Upper and lower bounds for the amout of thermal noise (guessed?)
Mscan = 2000
# Mscan = 200
noise_range = np.geomspace(0.005,5.0,num=Mscan)
# For block average
Mdn = 20

""" Parameters of the Langevin model """

# From other simulations and system preparation...
R0_drop = 15
etaw = 0.69
theta_e_md = 55.6
Lx_sub = 20.7
muf_md = 5.659896689453016

""" Finding optimal noise """

# Preparing the list to save opt. noise
opt_noise_xcl_vec = []
opt_noise_theta_vec = []

for n_sub in n_range :
    for a_sub in a_range :

        # Reference substrate
        # n_sub = 12
        # a_sub = 1.0
        n_sub_tag = str(n_sub).zfill(2)
        a_sub_tag = str(int(10*a_sub)).zfill(2)

        if MPI_RANK==MPI_ROOT :
            print("### --------- ###")
            print("### CASE "+n_sub_tag+a_sub_tag+" ###")
            print("### --------- ###")

        # Load MD simulation output
        md_output_an = md_output[n_sub_tag+a_sub_tag]

        # Loading MD data
        t_md = md_output_an['t']
        theta_avg_md = md_output_an['theta']
        x_avg_md = md_output_an['x']
        theta_std_md = md_output_an['theta_std']
        x_std_md = md_output_an['x_std']

        # Initial conditions
        a2 = a_sub**2
        rough_parameter = (2.0/np.pi) * np.sqrt(a2+1.0) * sc_spc.ellipe(a2/(a2+1.0))
        theta_g_0 = theta_avg_md[0]
        theta_g_0_flat_in = cos_m1(cos(theta_g_0)/rough_parameter)
        l_sub = Lx_sub/n_sub
        t_fin_md = t_md[-1]

        err_range_xcl = np.zeros(Mscan)
        err_range_theta = np.zeros(Mscan)

        ### Relative error on contact line position and contact angle ###

        for i in range(len(noise_range)) :
    
            if MPI_RANK==MPI_ROOT :
                print(">>> Simulation",(i+1),"/",Mscan)
                print(">>> Noise (nondim.),",noise_range[i])
    
            RS = RoughSubstrate(l=l_sub,mu_f=muf_md,R0=R0_drop,a=a_sub,theta_g_0_flat=theta_g_0_flat_in,
                            theta_e=theta_e_md,Gamma=noise_range[i])

            EM = EulerMurayama(RS=RS,t_fin=t_fin_md,t_bin=t_bin_em,M=m_rep)
            EM.simulate_ode(RS)
            EM.simulate_sde(RS)

            theta_w = RS.theta_w
            theta_fin_ode = EM.theta_g_vec[-1]
            theta_fin_sde = np.mean(EM.theta_g_ens[int(0.8*EM.Nt):])

            t_lang = RS.tau*EM.t_vec
            x_lang = TWOPI*EM.x_ens/RS.k
            x_lang_int = np.interp(t_md, t_lang, x_lang)
            theta_lang = EM.theta_g_ens
            theta_lang_int = np.interp(t_md, t_lang, theta_lang)

            err_xcl = np.sqrt((np.sum(x_avg_md-x_lang_int)**2))/(R0_drop*len(x_lang_int))
            err_theta = np.sqrt((np.sum(theta_avg_md-theta_lang_int)**2))/(theta_e_md*len(theta_lang_int))
            err_range_xcl[i] = err_xcl
            err_range_theta[i] = err_theta

            if MPI_RANK==MPI_ROOT :
                print(">>> Error on xcl (nondim.):",err_xcl)
                print(">>> Error on theta (nondim.):",err_theta)
                print("-----------------------------------------------------------------")

        # Block average
        noise_range_dn = np.mean(noise_range.reshape(-1, Mdn), axis=1)
        err_range_xcl_dn = np.mean(err_range_xcl.reshape(-1, Mdn), axis=1)
        err_range_theta_dn = np.mean(err_range_theta.reshape(-1, Mdn), axis=1)
        min_i_xcl = np.argmin(err_range_xcl_dn)
        min_i_theta = np.argmin(err_range_theta_dn)
        opt_noise_xcl = noise_range_dn[min_i_xcl]
        opt_noise_theta = noise_range_dn[min_i_theta]
        if MPI_RANK==MPI_ROOT :
            print("### ------------------------------------------------ ###")
            print("    Opt. noise xcl:",opt_noise_xcl)
            print("    Opt. noise theta:",opt_noise_theta)
            print("### ------------------------------------------------ ###")
        opt_noise_xcl_vec.append(opt_noise_xcl)
        opt_noise_theta_vec.append(opt_noise_theta)

        # Plotting relative errors
        if MPI_RANK==MPI_ROOT :
            fig1, ax1 = plt.subplots()
            ref_err_xcl = np.mean(x_std_md)/(5*R0_drop)
            ref_err_theta = np.mean(theta_std_md)/(5*theta_e_md)
            plt.loglog(noise_range_dn, err_range_xcl_dn, 'bo', label=r"$x_{cl}$")
            plt.loglog([noise_range_dn[0],noise_range_dn[-1]],[ref_err_xcl,ref_err_xcl],'b--',linewidth=2.5)
            plt.loglog(noise_range_dn, err_range_theta_dn, 'rs', label=r"$\theta$")
            plt.loglog([noise_range_dn[0],noise_range_dn[-1]],[ref_err_theta,ref_err_theta],'r--',linewidth=2.5)
            plt.xlabel(r'$\Gamma^*$ []',fontsize=AX_LABELS_FS)
            plt.ylabel('err []',fontsize=AX_LABELS_FS)
            plt.legend(fontsize=LEGEND_FS)
            plt.xticks(fontsize=AX_TICKS_FS)
            plt.yticks(fontsize=AX_TICKS_FS)
            plt.tight_layout()
            plt.savefig("figures/opt_noise_a"+a_sub_tag+"_n"+n_sub_tag+".png",format='png')
            # plt.show()

""" Saving optimal noise """
if MPI_RANK==MPI_ROOT :
    opt_noise_xcl_vec = np.array(opt_noise_xcl_vec)
    opt_noise_theta_vec = np.array(opt_noise_theta_vec)
    np.save("opt_noise_xcl_vec.npy", opt_noise_xcl_vec)
    np.save("opt_noise_theta_vec.npy", opt_noise_theta_vec)