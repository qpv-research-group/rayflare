# use pygmo DE to optimize thickness and doping level at front and back of cell, to
# match with measured values.

# load experimental data

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import seaborn as sns
from solcore.absorption_calculator import search_db
from solcore import material
from solcore.material_system import create_new_material
from solcore.structure import Layer
from rayflare.options import default_options
from rayflare.textures import regular_pyramids, planar_surface, heights_texture
from rayflare.ray_tracing import rt_structure

pal = sns.color_palette("husl", 3)

scan_width = 5.02 # in um

Lx = 1

# use amplitude images, but scale to ZSensor height?
# Not sure whether to use
AFM_pyramids_Z = np.loadtxt("data/Si_PyramidSide_30jan_0001_ZSensor.txt", skiprows=3)
AFM_alkaline_Z = np.loadtxt("data/si_AlkineEtchSide0003_ZSensor.txt", skiprows=3)

[front, back] = heights_texture(AFM_pyramids_Z*1e6, scan_width, scan_width, coverage_height=0)

P0 = front.P_0s
P1 = front.P_1s
P2 = front.P_2s
N = np.cross(P1 - P0, P2 - P0)

o_t = np.abs(np.real(np.arccos(N[:, 2] / (np.linalg.norm(N, axis=1)))))

h_dist = np.tan(o_t)*Lx/2

mean_h = np.mean(h_dist)
std_h = np.std(h_dist)

bins_ref = np.histogram(h_dist, bins=15, range=[mean_h-2*std_h, mean_h+2*std_h], density=True)

h_cent = (bins_ref[1][:-1] + bins_ref[1][1:])/2
probs = bins_ref[0]/np.sum(bins_ref[0])

[front, back] = heights_texture(AFM_alkaline_Z*1e6, scan_width, scan_width,
                                coverage_height=0)

P0 = front.P_0s
P1 = front.P_1s
P2 = front.P_2s
N = np.cross(P1 - P0, P2 - P0)

o_t_alk = np.abs(np.real(np.arccos(N[:, 2] / (np.linalg.norm(N, axis=1)))))

h_dist_alk = np.tan(o_t_alk)*Lx/2

# mean_h_alk = np.mean(h_dist)
# std_h_alk = np.std(h_dist)

bins_ref_alk = np.histogram(h_dist_alk, bins=20, range=[0, 0.7],
                         density=True)

h_cent_alk = (bins_ref_alk[1][:-1] + bins_ref_alk[1][1:])/2
probs_alk = bins_ref_alk[0]/np.sum(bins_ref_alk[0])

names = ["_alkaline.Sample.Raw", "_pyramid.Sample.Raw"]

path_RT = "data/R_T"
path_R = "data/R"

ind = [1, 2, 3]

alpha = [0.7, 1]

fig = plt.figure(figsize=(5.5, 3.7))

for i1, name in enumerate(names):

    for pref in ind:

        RT = np.loadtxt(path_RT + "/" + str(pref) + name + '.csv', skiprows=1, \
                                                                        delimiter=",")
        R = np.loadtxt(path_R + "/" + str(pref) + name + '.csv', skiprows=1,
                       delimiter=",")
        plt.plot(RT[:,0], 100-RT[:,1], label=pref, color=pal[pref-1], alpha=alpha[i1])
        plt.plot(R[:,0], R[:,1], '--', color=pal[pref-1], alpha=alpha[i1])

        T = RT[:, 1] - R[:,1]
        # T[T<0] = 0
        plt.plot(R[:, 0], T, '-.', color=pal[pref-1], alpha=alpha[i1])
        # plt.plot(R[:,0], 100-R[:,1]-T, '-', color=cols[i1])

plt.legend()
plt.show()

# use sample 2 for fitting purposes.
class fit_doping():

    def __init__(self, wl):
        # wl in nm
        RT_data_alk = np.loadtxt(path_RT + '/2_alkaline.Sample.Raw.csv',
                                 skiprows=1, delimiter=",")
        R_data_alk = np.loadtxt(path_R + '/2_alkaline.Sample.Raw.csv',
                                 skiprows=1, delimiter=",")

        RT_data_pyr = np.loadtxt(path_RT + '/2_pyramid.Sample.Raw.csv',
                                 skiprows=1, delimiter=",")
        R_data_pyr = np.loadtxt(path_R + '/2_pyramid.Sample.Raw.csv',
                                 skiprows=1, delimiter=",")

        A_data_alk = 100 - RT_data_alk
        A_data_pyr = 100 - RT_data_pyr

        self.wl = wl

        # self.wl = RT_data_alk[:,0][::-1] # in nm

        self.data = np.vstack((
            np.interp(wl, R_data_alk[:,0][::-1], R_data_alk[:,1][::-1]),
            np.interp(wl, R_data_alk[:,0][::-1], A_data_alk[:,1][::-1]),
            np.interp(wl, R_data_pyr[:,0][::-1], R_data_pyr[:,1][::-1]),
            np.interp(wl, R_data_pyr[:,0][::-1], A_data_pyr[:,1][::-1]),
                      )) # ascending wavelengths

        options = default_options()
        options.wavelengths = wl*1e-9
        options.pol = "u"
        options.I_thresh = 1e-3
        options.randomize_surface = True
        options.n_jobs = -1
        options.periodic = True
        options.project_name = "oxford_Si_doping_fitdoping"
        options.nx = 40
        options.ny = options.nx
        options.n_rays = 2 * options.nx ** 2
        options.lookuptable_angles = 500
        options.theta_in = 0

        self.options = options


    def calculate(self, x):

        # x: pyramid side doped layer thickness, doping level,
        # alkaline side doped layer thickness, doping level
        doping_pyr = np.round(x[1], 1)
        doping_alk = np.round(x[3], 1)
        th_pyr = x[0]
        th_alk = x[2]

        Si_doped_pyr = material("Si_FZ_Green_nk_1e" + str(doping_pyr))()
        Si_doped_alk = material("Si_FZ_Green_nk_1e" + str(doping_alk))()
        Si = material("Si")()
        SiOx = material("SiO")()
        Air = material("Air")()

        pyr_materials = [
            Layer(5e-9, SiOx),
            Layer(th_pyr*1e-9, Si_doped_pyr)
        ]

        alk_materials = [
            Layer(5e-9, SiOx),
            Layer(th_alk*1e-9, Si_doped_alk)
        ]

        front_pyramids = regular_pyramids(upright=True,
                                          elevation_angle=np.mean(o_t) * 180 / np.pi,
                                          height_distribution={"p": probs, "h": h_cent},
                                          interface_layers=pyr_materials,
                                          name="front_pyr")

        front_planar = regular_pyramids(
    upright=True,
    elevation_angle=h_cent_alk[np.argmax(probs_alk)]*180/np.pi,
    height_distribution={"p": probs_alk, "h": h_cent_alk},
    interface_layers=alk_materials, name="front_alk")

        back_pyramids = regular_pyramids(upright=False,
                                          interface_layers=pyr_materials[::-1],
                                         elevation_angle=np.mean(o_t) * 180 / np.pi,
                                         height_distribution={"p": probs, "h": h_cent},
                                          name="back_pyr")

        back_planar = regular_pyramids(interface_layers=alk_materials[::-1],
                                       upright=False,
                                       elevation_angle=h_cent_alk[np.argmax(
                                           probs_alk)] * 180 / np.pi,
                                       height_distribution={"p": probs_alk,
                                                            "h": h_cent_alk},
                                      name="back_alk")

        rtstr_alk = rt_structure(textures=[front_planar, back_pyramids],
                                      materials=[Si],
                                      widths=[150e-6],
                                      incidence=Air, transmission=Air,
                                      use_TMM=True, options=self.options,
                                      overwrite=True,
                                      )

        rtstr_pyr = rt_structure(textures=[front_pyramids, back_planar],
                                      materials=[Si],
                                      widths=[150e-6],
                                      incidence=Air, transmission=Air,
                                      use_TMM=True, options=self.options,
                                      overwrite=True,
                                      )

        result_alk = rtstr_alk.calculate(self.options)
        result_pyr = rtstr_pyr.calculate(self.options)

        total_A_alk = result_alk["A_per_layer"][:,0] + \
                      np.sum(result_alk["A_per_interface"][0], 1) + \
                        np.sum(result_alk["A_per_interface"][1], 1)

        total_A_pyr = result_pyr["A_per_layer"][:,0] + \
                        np.sum(result_pyr["A_per_interface"][0], 1) + \
                        np.sum(result_pyr["A_per_interface"][1], 1)
        # R_alk, A_alk, R_pyr, A_pyr
        result_stack = 100*np.vstack((result_alk["R"],
                                 total_A_alk,
                                 result_pyr["R"],
                                 total_A_pyr,
                                 )
                                     )

        # plt.figure()
        # plt.plot(self.wl, result_stack.T, '-k')
        # plt.plot(self.wl, self.data.T, '--r')
        # plt.show()

        return result_stack #, result_alk, result_pyr

    def fitness(self, x):

        rt_result = self.calculate(x)

        deviation = np.abs(self.data - rt_result)

        return [np.sum(deviation)]

    def get_bounds(self):
        return [[100, 18, 100, 18], [3000, 21, 3000, 21]]

# construct and save data for doped Si

# doping_levels = np.array([17, 18, 19, 20, 21])
#
# Si_wl = np.arange(250, 920.01, 2)*1e-9
#
# Si_wl_full = np.arange(250, 1450, 2)*1e-9
#
# Si_Green_2008 = search_db("Green-2008")[0][0]
#
# Si_PV_nk_mat = material(str(Si_Green_2008), nk_db=True)()
#
# Si_n_full = Si_PV_nk_mat.n(Si_wl_full)
#
# Si_PV_nk = np.array([Si_wl*1e6, Si_PV_nk_mat.k(Si_wl)]).T
#
# k_data_array = np.zeros((len(doping_levels), len(Si_wl_full)))
#
# plt.figure()
#
#
# for i1, doping in enumerate(doping_levels):
#     data = np.loadtxt("data/FZ_k_1e" + str(doping) + ".csv", delimiter=",")
#
#     data_k_constructed = np.vstack([Si_PV_nk, data])
#     k_data_array[i1, :] = np.interp(Si_wl_full*1e6, data_k_constructed[:, 0],
#                           data_k_constructed[:,1])
#
#     plt.semilogy(Si_wl_full*1e9, k_data_array[i1, :], '--k')
#
# doping_to_generate = np.arange(17, 21.01, 0.1)
#
# for i1, doping in enumerate(doping_to_generate):
#     doping = np.round(doping,1)
#     below = np.where(doping_levels == np.floor(doping))[0][0]
#     above = np.where(doping_levels == np.ceil(doping))[0][0]
#
#     k_interp = k_data_array[below, :] + \
#         (k_data_array[above, :] - k_data_array[below, :]) * \
#         (doping - np.floor(doping))
#     plt.semilogy(Si_wl_full*1e9, k_interp)
#
#     np.savetxt("data/FZ_Green_n_1e" + str(np.round(doping,1)) + "_interp.txt",
#                                            np.array([
#         Si_wl_full, Si_n_full]).T)
#
#     np.savetxt("data/FZ_Green_k_1e" + str(np.round(doping,1)) + "_interp.txt",
#                                            np.array([
#         Si_wl_full, k_interp]).T)
#
#     create_new_material("Si_FZ_Green_nk_1e" + str(doping),
#                         "data/FZ_Green_n_1e" + str(doping) + "_interp.txt",
#                         "data/FZ_Green_k_1e" + str(doping) + "_interp.txt",
#                         overwrite=True)
#
#
#
# plt.xlim(900, 1300)
# plt.show()

# test = fit_doping(np.linspace(1150, 1300, 20))
# # res = test.calculate([1000, 21, 1000, 21])
# print(test.fitness([1000, 21, 1000, 21]))
# print(test.fitness([500, 19, 500, 19]))


p_init = fit_doping(np.linspace(1150, 1300, 10))

prob = pg.problem(p_init)
algo = pg.algorithm(
    pg.de(
        # gen=1000*n_juncs,
        gen=10,
        F=1,
        CR=1,
        xtol=0.1,
        ftol=0.1,
    )
)

pop = pg.population(prob, 20)
pop = algo.evolve(pop)






