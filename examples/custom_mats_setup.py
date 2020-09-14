from solcore.config_tools import reset_defaults
from solcore.material_system import create_new_material
from solcore.config_tools import add_source
import numpy as np
import os

# IMPORTANT: if you have not used custom materials or the refractiveindex.info database in Solcore before it is possible
# you do not have a hidden .solcore_config file in your home directory. In that case, run this, and you should be prompted
# to create one. If you have already added custom materials, running reset_defaults will reset your config file and you will
# need to re-add them to the database so be careful

# reset_defaults()

# To set up the use of custom materials, first we need to tell Solcore some things about where to put custom materials. for this,
# we use the add_source function from config_tools, although we could also manually edit
# the solcore configuration file (which should be in your home directory).
# You need to add two things to the config file: where to put the n and k data for new
# materials added to the database, and where to put the other parameters (these can all
# go in the same file).

home_folder = os.path.expanduser('~')
custom_nk_path = os.path.join(home_folder, 'Solcore/custommats')
nk_db_path = os.path.join(home_folder, 'Solcore/NK.db')
param_path = os.path.join(home_folder, 'Solcore/custom_params.txt')

add_source('Others', 'custom_mats', custom_nk_path)
add_source('Others', 'nk', nk_db_path)
add_source('Parameters', 'custom', param_path)

import pandas as pd
names = ['GaAs_subs', 'InAlP', 'InGaP', 'SiGeSn_29111']

for name in names[:3]:
    data = pd.read_csv('data/'+name+'.txt', sep='\t', names=['wl', 'n', 'k'])
    data['wl'] = data['wl']*1e-9
    np.savetxt('data/'+name+'_n.txt', data[['wl', 'n']])
    np.savetxt('data/'+name+'_k.txt', data[['wl', 'k']])


sol_names = ['GaAs_WVASE', 'InAlP_WVASE', 'InGaP_WVASE', 'SiGeSn']

for i1, name in enumerate(sol_names):
    create_new_material(name, 'data/'+names[i1]+'_n.txt', 'data/'+names[i1]+'_k.txt')