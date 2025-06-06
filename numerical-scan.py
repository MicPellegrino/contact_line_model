import numpy as np
from production import productionRun

# Defining baseline parameters
R0_md = 20
theta_e_md = 55.6
t_fin_md = 15
M = 56

# Colormap value boundaries
cah_plot_cutoff = None
clf_plot_cutoff = None

# Define range of cl friction and noise
noise_vec = np.linspace(0.1,0.5,5)
cl_friction_vec = np.linspace(4,8,5)

for i in range(len(noise_vec)) :
    for j in range(len(cl_friction_vec)) :
        productionRun(noise=noise_vec[i], cl_friction=cl_friction_vec[j],
            R0_md=R0_md, theta_e_md=theta_e_md, t_fin_md=t_fin_md,M=M,
            cah_plot_cutoff=cah_plot_cutoff, clf_plot_cutoff=clf_plot_cutoff)