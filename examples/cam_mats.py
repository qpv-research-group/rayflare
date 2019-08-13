import numpy as np
import matplotlib.pyplot as plt

from solcore import si, material
from solcore.structure import Junction, Layer
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver, default_options
from solcore.light_source import LightSource
from rigorous_coupled_wave_analysis.rcwa import rcwa
from solcore.absorption_calculator.nk_db import download_db, search_db
from solcore.material_system.create_new_material import create_new_material
import xarray as xr

from angles import make_angle_vector, theta_summary
calc = True

import pandas as pd
names = ['GaAs_subs', 'InAlP', 'InGaP']

for name in names:
    data = pd.read_csv('data/'+name+'.txt', sep='\t', names=['wl', 'n', 'k'])
    data['wl'] = data['wl']*1e-9
    np.savetxt('data/'+name+'_n.txt', data[['wl', 'n']])
    np.savetxt('data/'+name+'_k.txt', data[['wl', 'k']])


sol_names = ['GaAs_WVASE', 'InAlP_WVASE', 'InGaP_WVASE']

for i1, name in enumerate(sol_names):
    create_new_material(name, 'data/'+names[i1]+'_n.txt', 'data/'+names[i1]+'_k.txt')