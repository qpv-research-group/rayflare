from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
from rayflare.textures import xyz_texture
from time import time
import os
import seaborn as sns
from rayflare.options import default_options

options = default_options()
nx = 70

thetas = np.linspace(0, np.pi / 2 - 0.05, 100)


thetas_1 = thetas[0:25]
thetas_2 = thetas[25:50]
thetas_3 = thetas[50:75]
thetas_4 = thetas[75:100]


thetas_min = []
thetas_max = []

for theta_g in [thetas_1, thetas_2, thetas_3, thetas_4]:

    thetas_min.append(np.int(10*np.round(180*np.min(theta_g)/np.pi, 1)))
    thetas_max.append(np.int(10*np.round(180*np.max(theta_g)/np.pi, 1)))

T_v = []
T_t = []
n_int = []
theta_dist = []



for i1 in range(4):

    theta_min = thetas_min[i1]
    theta_max = thetas_max[i1]

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2
    options.xlim = 0.05
    options.ylim = 0.05
    
    options.theta = 0.1
    options.nx = nx
    options.ny = nx


    if os.path.isfile('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '45_2.txt'):

        T_values = np.loadtxt('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '45_2.txt')
        T_total = np.loadtxt('results/sphere_raytrace_totalT_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '45_2.txt')
        n_interactions = np.loadtxt('results/sphere_raytrace_ninter_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '45_2.txt')
        theta_distribution = np.loadtxt('results/sphere_raytrace_thetas_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '45_2.txt')

        T_v.append(T_values)
        T_t.append(T_total)
        n_int.append(n_interactions)
        theta_dist.append(theta_distribution)


T_all = np.hstack(T_v)
T_tot_all = np.hstack(T_t)
n_int_all = np.hstack(n_int)
th_dist_all = np.vstack(theta_dist)

minimum_angle = np.pi - 44.5*np.pi/180

min_30 = np.pi - 30*np.pi/180
min_25 = np.pi - 25*np.pi/180
min_20 = np.pi - 20*np.pi/180

min_angle_old = np.pi - 17.5 * np.pi / 180

T_175 = np.array([np.sum(x > min_angle_old) / options.n_rays for x in th_dist_all])
T_45_2 = np.array([np.sum(x > minimum_angle) / options.n_rays for x in th_dist_all])
T_30 = np.array([np.sum(x > min_30) / options.n_rays for x in th_dist_all])
T_25 = np.array([np.sum(x > min_25) / options.n_rays for x in th_dist_all])
T_20 = np.array([np.sum(x > min_20) / options.n_rays for x in th_dist_all])

cols = sns.color_palette('rocket', 6)

plt.figure()
plt.plot(thetas*180/np.pi, T_tot_all, color=cols[0], label=r'$\theta_c = 90^\circ$', alpha=0.5)
plt.scatter(thetas*180/np.pi, T_tot_all, edgecolors=cols[0], facecolors='none', alpha=0.5)
plt.plot(thetas*180/np.pi, T_45_2, color=cols[1], label=r'$\theta_c = 44.5^\circ$')
plt.scatter(thetas*180/np.pi, T_45_2, edgecolors=cols[1], facecolors='none')
plt.plot(thetas*180/np.pi, T_30, color=cols[2], label=r'$\theta_c = 30^\circ$')
plt.scatter(thetas*180/np.pi, T_30, edgecolors=cols[2], facecolors='none')
plt.plot(thetas*180/np.pi, T_25, color=cols[3], label=r'$\theta_c = 25^\circ$')
plt.scatter(thetas*180/np.pi, T_25, edgecolors=cols[3], facecolors='none')
plt.plot(thetas*180/np.pi, T_20, color=cols[4], label=r'$\theta_c = 20^\circ$')
plt.scatter(thetas*180/np.pi, T_20, edgecolors=cols[4], facecolors='none')
plt.plot(thetas*180/np.pi, T_175, color=cols[5], label=r'$\theta_c = 17.5^\circ$')
plt.scatter(thetas*180/np.pi, T_175, edgecolors=cols[5], facecolors='none')
# plt.legend(title="Number of rays")
plt.xlim(0, 90)
plt.xlabel(r'$\beta$ (rads)')
plt.ylabel('Transmission')
plt.legend()


# plt.title(r'Convergence with number of rays ($N_{triangles} = 2^{12}$)')

plt.show()

np.savetxt('hyperhemi_T_withinangle_u.txt', np.vstack([thetas, T_all, T_tot_all]).T)
