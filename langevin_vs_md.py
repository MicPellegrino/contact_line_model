import numpy as np
import scipy.special as sc_spc

from comparison import RoughSubstrate, EulerMurayama
from comparison import TWOPI, MPI_COMM, MPI_RANK, MPI_SIZE, MPI_ROOT

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

AX_LABELS_FS = 25
AX_TICKS_FS = 20
LEGEND_FS = 22.5

# Some useful definitions...
cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

# Load MD simulation output
"""
n_tags = ['06','08','10','12','14','16']
n_range = [6,8,10,12,14,16]
md_output = dict()
for nt in n_tags :
    md_output[nt] = np.load("MD_DATA/data_a10_n"+nt+".npz")
"""

# Reference substrate
n_sub = 16
n_sub_tag = str(n_sub).zfill(2)
a_sub = 1.0
a_sub_tag = str(int(10*a_sub)).zfill(2)

# Found by "visual inspection" for a given substrate (see calibrate_noise.py)
noise_opt = 0.5

# Load MD simulation output
md_output = np.load("MD_DATA/data_a"+a_sub_tag+"_n"+n_sub_tag+".npz")

# Loading MD data
t_md = md_output['t']
theta_avg_md = md_output['theta']
x_avg_md = md_output['x']
theta_std_md = md_output['theta_std']
x_std_md = md_output['x_std']

# From other simulations and system preparation...
R0_drop = 15
etaw = 0.69
theta_e_md = 55.6
Lx_sub = 20.7
l_sub = Lx_sub/n_sub
muf_md = 5.659896689453016
t_fin_md = t_md[-1]

# Initial conditions
a2 = a_sub**2
rough_parameter = (2.0/np.pi) * np.sqrt(a2+1.0) * sc_spc.ellipe(a2/(a2+1.0))
theta_g_0 = theta_avg_md[0]
theta_g_0_flat_in = cos_m1(cos(theta_g_0)/rough_parameter)

RS = RoughSubstrate(l=l_sub,mu_f=muf_md,R0=R0_drop,a=a_sub,theta_g_0_flat=theta_g_0_flat_in,
                    theta_e=theta_e_md,Gamma=noise_opt)

# Number of replicates for the Langevin model
m_rep = 56
# Time-step for the EM method
t_bin_em = 0.1

EM = EulerMurayama(RS=RS,t_fin=t_fin_md,t_bin=t_bin_em,M=m_rep)
EM.simulate_ode(RS)
EM.simulate_sde(RS)

theta_w = RS.theta_w

""" PLOTTING """

x_avg_lg = TWOPI*EM.x_ens/RS.k
x_std_lg = TWOPI*EM.x_std/RS.k
t_lg = RS.tau*EM.t_vec

if MPI_RANK == MPI_ROOT :

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t_md, x_avg_md, 'k-', linewidth=3.25)
    ax1.fill_between(t_md,x_avg_md-x_std_md,x_avg_md+x_std_md,color='0.5',alpha=0.25,linewidth=0.0)

    ax1.plot(t_lg, x_avg_lg, 'r-', linewidth=3.0)
    ax1.plot(t_lg, x_avg_lg+x_std_lg, 'r:', linewidth=1.25)
    ax1.plot(t_lg, x_avg_lg-x_std_lg, 'r:', linewidth=1.25)
    # ax1.fill_between(t_lg, x_avg_lg-x_std_lg, x_avg_lg+x_std_lg,
    #     color='r',alpha=0.25,linewidth=0.0)

    ax1.set_ylabel(r'$x_{cl}$ [nm]', fontsize=AX_LABELS_FS)
    ax1.set_xlim([t_lg[0], t_lg[-1]])
    ax1.tick_params(axis='x',which='both',labelbottom=False)
    ax1.tick_params(axis='y', labelsize=AX_TICKS_FS)

    ax2.plot(t_md, theta_avg_md, 'k-', linewidth=3.25, label='MD')
    ax2.fill_between(t_md,theta_avg_md-theta_std_md,theta_avg_md+theta_std_md,color='0.5',alpha=0.25,linewidth=0.0)

    ax2.plot(t_lg, EM.theta_g_ens, 'r-', linewidth=3, label='Langevin')
    ax2.plot(t_lg, EM.theta_g_ens+EM.theta_std, 'r:', linewidth=1.25)
    ax2.plot(t_lg, EM.theta_g_ens-EM.theta_std, 'r:', linewidth=1.25)
    ax2.plot(t_lg, RS.theta_w*np.ones(EM.t_vec.shape), 'g--', linewidth=3.25, label=r'$\theta_W$')

    ax2.set_xlabel(r'$t$ [ns]', fontsize=AX_LABELS_FS)
    ax2.set_ylabel(r'$\theta_g$ [deg]', fontsize=AX_LABELS_FS)
    ax2.set_xlim([t_lg[0], t_lg[-1]])
    ax2.legend(fontsize=LEGEND_FS)
    ax2.tick_params(axis='x', labelsize=AX_TICKS_FS)
    ax2.tick_params(axis='y', labelsize=AX_TICKS_FS)

    plt.tight_layout()
    plt.show()