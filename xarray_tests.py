import numpy as np
import pandas as pd
import xarray as xr

n_w = 10 # wavelengths
n_a = 2600 # angles

n_l = 3 # surface layers - different for every surface

mat = xr.DataArray(np.abs(np.random.random(n_w, n_a, n_a)),
                   [('wl', np.linspace(200, 500, n_w)),
                    ('a_out', np.linspace(0, np.pi, n_a)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

mat_abs = xr.DataArray(np.abs(np.random.randn(n_w, n_l, n_a)),
                   [('wl', np.linspace(200, 500, n_w)),
                    ('layer', np.arange(n_l)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

vec = xr.DataArray(np.abs(np.random.randn(n_w, n_a)), [('wl', np.linspace(200, 500, n_w)),
                                                             ('a_in', np.linspace(0, np.pi, n_a))])


# correct multiplication: np.dot(mat.sel(wl=250), vec.sel(wl=250))

mat.dot(vec, dims = ['a_in']).rename({'a_out': 'a_in'})

def dot_wl(mat, vec):
    return(mat.dot(vec, dims=['a_in']).rename({'a_out': 'a_in'}))

def sum_wl(vec):
    return(vec.sum(dim = ['a_in']))
#vfront: incoming vector

vfront_prime = dot_wl(A_n, vec)

# iteration: order is D -> C -> D -> B ... repeat
# + the absorption matrices

vback_i = dot_wl(D_n, vfront_prime)
vback_prime_i = dot_wl(C_n, vback)
vfront_i = dot_wl(D_n, vback_prime)
vfront_prime_i = dot_wl(B_n, vfront)

R_i = vfront_i.reduce(np.sum, dim = ['a_in']) - vfront_prime_i.reduce(np.sum, dim = ['a_in']) # not angle-resolved
T_i = vback_i.reduce(np.sum, dim = ['a_in']) - vback_prime_i.reduce(np.sum, dim = ['a_in'])
A_i = vfront_prime_i.reduce(np.sum, dim = ['a_in']) - vback_i.reduce(np.sum, dim = ['a_in']) \
    + vback_prime_i.reduce(np.sum, dim = ['a_in']) - vfront_i.reduce(np.sum, dim = ['a_in'])

R_i = sum_wl(vfront_i.reduce) - sum_wl(vfront_prime_i) # not angle-resolved
T_i = sum_wl(vback_i.reduce) - sum_wl(vback_prime_i)
Abulk_i = sum_wl(vfront_prime_i) - sum_wl(vback_i.reduce) \
    + sum_wl(vback_prime_i) - sum_wl(vfront_i)

vtrans_i = dot_wl(A_nplus, vback_i)

import numpy as np
import pandas as pd
import xarray as xr

n_w = 10 # wavelengths
n_a = 2600 # angles

for i1 in np.arange(20):
    mat1 = xr.DataArray(np.abs(np.random.randn(n_w, n_a, n_a)),
                   [('wl', np.linspace(i1, i1+1-0.01, n_w)),
                    ('a_out', np.linspace(0, np.pi, n_a)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

    mat1.to_netcdf(path='test' + str(i1) + '.nc', mode='w')

mat2 = xr.DataArray(np.abs(np.random.randn(n_w, n_a, n_a)),
                   [('wl', np.linspace(301, 400, n_w)),
                    ('a_out', np.linspace(0, np.pi, n_a)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

mat2.to_netcdf(path='netCDFtest2.c', mode='w')

mat3 = xr.DataArray(np.abs(np.random.randn(n_w, n_a, n_a)),
                   [('wl', np.linspace(401, 500, n_w)),
                    ('a_out', np.linspace(0, np.pi, n_a)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

mat3.to_netcdf(path='netCDFtest3.nc', mode='w')

mat4 = xr.DataArray(np.abs(np.random.randn(n_w, n_a, n_a)),
                   [('wl', np.linspace(501, 600, n_w)),
                    ('a_out', np.linspace(0, np.pi, n_a)),
                    ('a_in', np.linspace(0, np.pi, n_a))])

mat4.to_netcdf(path='netCDFtest4.nc', mode='w')

xr.open_dataset('netCDFtest')

# xarray.open_mfdataset

a = xr.open_mfdataset('C:/Users/pmpea/Box Sync/Optics package/*.nc',
                      chunks = {'wl': 200})

from scipy.sparse import csc_matrix, save_npz

mat = abs(np.random.random((n_a, n_a)))
mat[mat < 0.99] = 0

mat_sparse = csc_matrix(mat)
np.save('matsave', mat)
save_npz('sparsematsave.npz', mat_sparse)