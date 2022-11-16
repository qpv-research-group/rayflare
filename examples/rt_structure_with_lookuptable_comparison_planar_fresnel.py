import numpy as np
import os

from solcore.structure import Layer
from solcore import material, si
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import regular_pyramids, planar_surface
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
from rayflare.transfer_matrix_method import tmm_structure

from rayflare.angles import make_angle_vector

import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

elevation_angle = 55

wavelengths = np.linspace(1000, 1200, 18) * 1e-9

n_thetas = [10, 20, 60, 100, 200, 300]
# n_thetas = [10, 20]
cols = sns.color_palette("husl", len(n_thetas))
plt.figure()

# thoughts: check theta distribution at the end, anything interesting?

for j1, nth in enumerate(n_thetas):

    options = default_options()
    options.wavelengths = wavelengths
    # options.n_theta_bins = 60
    # options.nx = 10
    # options.ny = options.nx
    # options.n_rays = 4*480*options.nx**2
    options.n_theta_bins = nth

    _, _, av = make_angle_vector(options.n_theta_bins, options.phi_symmetry, options.c_azimuth)
    options.nx = 10
    options.ny = options.nx
    options.n_rays = (len(av)/2)*options.nx**2
    options.depth_spacing = 1e-9
    options.pol = "s"
    options.I_thresh = 1e-3
    options.only_incidence_angle = True
    options.theta_in = 0
    # options.theta_spacing = "linear"
    options.project_name = "testing_fr_air_3" + str(elevation_angle) + str(len(av))
    #

    Si = material("Si")()
    Air = material("Air")()
    MgF2 = material("MgF2_RdeM")()
    ITO_back = material("ITO_lowdoping")()
    Perovskite = material("Perovskite_CsBr_1p6eV")()
    Ag = material("Ag_Jiang")()
    aSi_i = material("aSi_i")()
    aSi_p = material("aSi_p")()
    aSi_n = material("aSi_n")()
    LiF = material("LiF")()
    IZO = material("IZO")()
    C60 = material("C60")()

    # stack based on doi:10.1038/s41563-018-0115-4
    front_materials = [
        Layer(100e-9, MgF2),
        Layer(110e-9, IZO),
        Layer(15e-9, C60),
        Layer(1e-9, LiF),
        Layer(440e-9, Perovskite),
        Layer(6.5e-9, aSi_n),
        Layer(6.5e-9, aSi_i),
    ]

    back_materials = [
        Layer(6.5e-9, aSi_i),
        Layer(6.5e-9, aSi_p),
        Layer(240e-9, ITO_back)
    ]


    # whether pyramids are upright or inverted is relative to front incidence.
    # so if the same etch is applied to both sides of a slab of silicon, one surface
    # will have 'upright' pyramids and the other side will have 'not upright' (inverted)
    # pyramids in the model

    surf = regular_pyramids(elevation_angle=elevation_angle, size=1, upright=True)
    surf_back = regular_pyramids(elevation_angle=elevation_angle, size=1, upright=False)

    front_surf = Interface(
        "RT_Fresnel",
        texture=surf,
        # layers=front_materials,
        name="Perovskite_aSi_coherent",
        # coherent=True,
        # coherency_list = ["i"]*len(front_materials)
    )

    back_surf = Interface(
        "RT_Fresnel", texture=surf_back,
        #layers=back_materials,
        name="aSi_ITO_coherent", #coherent=True
    )


    bulk_Si = BulkLayer(260e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

    process_structure(SC, options, save_location="current")

    result = calculate_RAT(SC, options, save_location="current")
    result_ARM = result[0]
    result_ARM_perpass = result[1]


    results_per_layer_front = np.sum(result_ARM_perpass["a"][0], 0)
    results_per_layer_back = np.sum(result_ARM_perpass["a"][1], 0)


    plt.subplot(1,1,1)
    plt.plot(wavelengths*1e9, result_ARM['R'][0], '-', label = str(nth), color=cols[j1])
    plt.plot(wavelengths*1e9, result_ARM['T'][0], '--', color=cols[j1])
    plt.plot(wavelengths*1e9, result_ARM['A_bulk'].T, '-.', color=cols[j1])
    # plt.plot(wavelengths*1e9, results_per_layer_front, '-o')
    # plt.plot(wavelengths*1e9, results_per_layer_back, '-o')
    # plt.plot(wavelengths*1e9, result_TMM["A_per_layer"][:,len(front_materials)], '--b')

options.n_rays = 2000

# ray-tracing WITHOUT angular redistribution method

triangle_surf = regular_pyramids(elevation_angle=elevation_angle, upright=True, size=1,
                                 # interface_layers=front_materials,
                                 # name="coh_front"
                                 )

triangle_surf_back = regular_pyramids(elevation_angle=elevation_angle, upright=False, size=1,
                                      # interface_layers=back_materials,
                                      # name="coh_back"
                                      )

rtstr_inc = rt_structure(textures=[triangle_surf, triangle_surf_back],
                         materials=[Si],
                         widths=[260e-6],
                         incidence=Air, transmission=Air,
                         # use_TMM=True, options=options, save_location="current"
                         )
# options.parallel = False
result_RT = rtstr_inc.calculate(options)
plt.plot(wavelengths*1e9, result_RT['R'], '-ko', mfc='none')
plt.plot(wavelengths*1e9, result_RT['T'], '--ko', mfc='none')
plt.plot(wavelengths*1e9, result_RT['A_per_layer'][:,0], '-.ko', mfc='none')
# plt.plot(wavelengths*1e9, result_RT['A_per_interface'][0], '--o', mfc='none')
# plt.plot(wavelengths*1e9, result_RT['A_per_interface'][1], '--o', mfc='none')

plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.legend()
plt.ylim(0, 1)
plt.title(str(options.n_theta_bins))
plt.show()


# fig=plt.figure(figsize=(9,3.7))
# plt.subplot(1,1,1)
# plt.plot(wavelengths*1e9, result_ARM['R'][0], 'k-o')
# plt.plot(wavelengths*1e9, result_ARM['T'][0], 'r-o')
#
# plt.plot(wavelengths*1e9, result_RT['R'], 'k--o', mfc='none')
# plt.plot(wavelengths*1e9, result_RT['T'], 'r--o', mfc='none')
#
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('R / A / T')
# plt.xlim(300, 1200)
#
# plt.show()

thetadist = np.sum(result_ARM_perpass['r'][0],0)

thetadist_wl = thetadist[0]

_, _, av = make_angle_vector(options.n_theta_bins, options.phi_symmetry, options.c_azimuth)

angledist = np.zeros((len(np.unique(av[:,1])), 2))
angledist[:,0] = np.unique(av[:,1])

for i1 in range(len(thetadist_wl)):
    angledist[angledist[:,0] == av[i1,1], 1] += thetadist_wl[i1]

angledist = angledist[:int(len(angledist)/2), :]

plt.hist(angledist[:,1], angledist[:,0])
plt.show()


RT_dist = result_RT["thetas"][0]
RT_dist = RT_dist[~np.isnan(RT_dist)]
RT_dist = RT_dist[RT_dist < np.pi/2]

plt.hist(RT_dist)
plt.show()