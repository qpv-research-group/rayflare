import numpy as np
import xarray as xr
from sparse import load_npz, dot, tensordot, COO, stack
from config import results_path
from angles import make_angle_vector
from solcore import material
import os

theta_intv, phi_intv, angle_vector = make_angle_vector(100, np.pi/2, 0.25)
n_a_in = int(len(angle_vector)/2)

Si = material('Si')()
num_wl = 2

wls = np.linspace(700, 800, num_wl)*1e-9
alphas = Si.alpha(wls)

# bulk thickness in m
thick = 1000*1e-6

thetas = angle_vector[:n_a_in, 1]

def make_D(alphas, thick, thetas):
    diag = np.exp(-alphas[:, None] * thick / np.cos(thetas[None, :]))
    D_1 = stack([COO(np.diag(x)) for x in diag])
    return D_1

D_1 = make_D(alphas, thick, thetas)

mat_path = os.path.join(results_path, 'testing', 'GeGaAsstackRT.npz')
absmat_path = os.path.join(results_path, 'testing', 'GeGaAsstackA.npz')

v0 = np.zeros((num_wl, n_a_in))
v0[:,0] = 1

fullmat = load_npz(mat_path)
absmat = load_npz(absmat_path)

Rf_1 = fullmat[:, :n_a_in, :]
Tf_1 = fullmat[:, n_a_in:, :]
Af_1 = absmat

Rb_1 = fullmat[:, :n_a_in,:]
Tb_1 = fullmat[:, n_a_in:, :]
Ab_1 = absmat

Rf_2 = fullmat[:, :n_a_in,:]
Tf_2 = fullmat[:, n_a_in:, :]
Af_2 = absmat

print(np.sum(Rf_1[0].todense(), 0))
print(np.sum(Tf_1[0].todense(), 0))
print(np.sum(Af_1[0].todense(), 0))


# v0.groupby('wl').apply(lambda x: dot(R, x.data))

def dot_wl(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1]))
    for i1 in range(vec.shape[0]):
        result[i1, :] = dot(mat[i1, :], vec[i1])
    return result



vf_1 = dot_wl(Tf_1, v0) # pass through front surface
vr_1 = dot_wl(Rf_1, v0) # reflected from front surface
a_1 = dot_wl(Af_1, v0) # absorbed in front surface at first interaction

# rep
vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
vf_3 = dot_wl(Rb_1, vf_2) # reflect from front surface

# other results:
vr_3 = dot_wl(Tb_1, vf_2) # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
vt_2 = dot_wl(Tf_2, vb_1) # transmitted into medium below through back surface
a_2 = dot_wl(Af_2, vb_1) # absorbed in 2nd surface
a_1 = dot_wl(Ab_1, vf_2) # absorbed in 1st surface (from the back)


# end rep



