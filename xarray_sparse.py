import numpy as np
import xarray as xr
import sparse
import dask.array as da

n_w = 2 # wavelengths
n_a = 2600 # angles

n_l = 3 # surface layers - different for every surface

x = np.random.random((n_a, n_a))
x[x < 0.99] = 0  # fill most of the array with zeros

s = sparse.COO(x)  # convert to sparse array

sparse.save_npz('sparsematsave2.npz', s)