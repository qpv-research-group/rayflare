import numpy as np
from solcore.structure import Layer
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from solcore import material
from solcore.material_system import create_new_material
from solcore.absorption_calculator.nk_db import create_nk_txt, search_db

from structure import Interface, RTgroup, BulkLayer, Structure
from ray_tracing.rt_lookup import RTSurface, RT
import math
from matrix_formalism.multiply_matrices import matrix_multiplication
from time import time

import matplotlib.pyplot as plt

### structure ###
# Air
# ---- pyramids: layers are GaInP, GaAs
# Si
# ---- pyramids: layers are GaAs
# Air

# matrix multiplication
wavelengths = np.linspace(300, 1150, 800)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'testing2',
           'calc_profile': True,
           'n_theta_bins': 25,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 0, 'phi_in': 0,
           'I_thresh': 1e-4,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 1000,
           'prof_layers': [1,2],
           'n_rays': 50000,
           'random_angles': False,
           'nx': 2, 'ny': 2,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2}

#create_new_material('Perovskite_CsBr', 'examples/CsBr10p_1to2_n.txt', 'examples/CsBr10p_1to2_k.txt')
#create_new_material('ITO_lowdoping', 'examples/model_back_ito_n.txt', 'examples/model_back_ito_k.txt')
#create_new_material('Ag_Jiang', 'examples/Ag_UNSW_n.txt', 'examples/Ag_UNSW_k.txt')
#create_new_material('aSi_i', 'examples/model_i_a_silicon_n.txt', 'examples/model_i_a_silicon_k.txt')
#create_new_material('aSi_p', 'examples/model_p_a_silicon_n.txt', 'examples/model_p_a_silicon_k.txt')
#create_new_material('aSi_n', 'examples/model_n_a_silicon_n.txt', 'examples/model_n_a_silicon_k.txt')
#create_new_material('MgF2_RdeM', 'examples/MgF2_RdeM_n.txt', 'examples/MgF2_RdeM_k.txt')
#create_nk_txt('195', 'LiF', 'examples')
#create_new_material('LiF', 'examples/LiF_n.txt', 'examples/LiF_k.txt')
Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()
LiF = material('LiF')()
IZO = material('IZO')()
C60 = material('C60')()
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)

front_materials = [Layer(100e-9, MgF2), Layer(110e-9, IZO),
                   SnO2,
                   Layer(15e-9, C60),
                   Layer(1e-9, LiF),
                   Layer(440e-9, Perovskite),
                   Spiro,
                   Layer(6.5e-9, aSi_n), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_LT',
                       coherent=True, prof_layers = [6])
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_LT',
                      coherent=True, prof_layers = [1,2,3])

bulk_Si = BulkLayer(260e-6, Si, name = 'Si_bulk')

SC = Structure([front_surf, bulk_Si, back_surf], incidence = Air, transmission = Ag)

# making lookuptable

for i1, struct in enumerate(SC):
    if type(struct) == Interface:
        # is this an interface type which requires a lookup table?
        if struct.method == 'RT_TMM':
            print('Making lookuptable for element ' + str(i1) + ' in structure')
            if i1 == 0:
                incidence = SC.incidence
            else:
                incidence = SC[i1-1].material # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission

            else:
                substrate = SC[i1+1].material # bulk material below
            print(incidence)
            print(substrate)
            coherent = struct.coherent
            if not coherent:
                coherency_list = struct.coherency_list
            else:
                coherency_list = None
            prof_layers = struct.prof_layers

            allres = make_TMM_lookuptable(struct.layers, substrate, incidence, struct.name,
                                          options, coherent, coherency_list, prof_layers)

            allres_plot = allres.assign_coords(angle= allres.coords['angle']*180/np.pi)
            fig1 = plt.figure()
            labels = ['a) R (front)', 'b) T (front)', 'c) R (back)', 'd) T (back)']
            lab_ind = 0
            for j1, side in enumerate([1, -1]):
                ax1 = fig1.add_subplot(2, 2, j1+1)
                allres_plot['R'].sel(side=side, pol='u').plot.imshow('wl', 'angle', ax = ax1,
                                                                     vmin=0, vmax=1.001)
                print(np.max(allres_plot['R'].sel(side=side, pol='u')))
                if side == 1:
                    ax1.set_ylabel('Angle (°)')
                else:
                    ax1.set_ylabel('')
                ax1.set_xlabel('')
                
                ax1.set_title(labels[lab_ind])
                lab_ind +=1
                ax1.yaxis.set_ticks([0, 45, 90])
                ax2 = fig1.add_subplot(2, 2, j1+3)
                allres_plot['T'].sel(side=side, pol='u').plot.imshow('wl', 'angle', ax = ax2,
                                                                     vmin=0, vmax=1.001)
                ax2.set_xlabel('Wavelength (nm)')
                if side == 1:
                    ax2.set_ylabel('Angle (°)')
                else:
                    ax2.set_ylabel('')
                ax2.set_title(labels[lab_ind])
                lab_ind += 1
                ax2.yaxis.set_ticks([0, 45, 90])

            fig2 = plt.figure()
            for j1, side in enumerate([1, -1]):

                for i1, layer in enumerate(prof_layers):
                    ax = fig2.add_subplot(len(prof_layers), 2, 2*(i1) + j1+1)
                    allres_plot['Alayer'].sel(side=side,
                                              pol='u', layer=layer).plot.imshow('wl', 'angle', ax=ax,
                                                                     vmin=0, vmax=1.001)
                    ax.set_xlabel('')
                    if side == 1:
                        ax.set_ylabel('Angle (°)')
                    else:
                        ax.set_ylabel('')
                    #ax.set_title('')
                    ax.yaxis.set_ticks([0, 45, 90])
                ax.set_xlabel('Wavelength (nm)')
                plt.show()

