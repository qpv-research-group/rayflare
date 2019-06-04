import numpy as np
import xarray as xr
from sparse import load_npz, dot, tensordot, COO, stack
from config import results_path
from angles import make_angle_vector
from solcore import material
import os
import matplotlib.pyplot as plt
import xarray as xr
from time import time

lookuptable = xr.open_dataset('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsonSi.nc')

depth = 500

theta_intv, phi_intv, angle_vector = make_angle_vector(100, np.pi/2, 0.25)
n_a_in = int(len(angle_vector)/2)

Si = material('Si')()
num_wl = 4

wls = np.linspace(600, 1100, num_wl)*1e-9
alphas = Si.alpha(wls)

# bulk thickness in m
thick = 0.5*1e-6

thetas = angle_vector[:n_a_in, 1]

def make_D(alphas, thick, thetas):
    diag = np.exp(-alphas[:, None] * thick / np.cos(thetas[None, :]))
    D_1 = stack([COO(np.diag(x)) for x in diag])
    return D_1

D_1 = make_D(alphas, thick, thetas)

v0 = np.zeros((num_wl, n_a_in))
v0[:,0] = 1

# front incidence matrices (both back and rear surface)
mat_path = os.path.join(results_path, 'testing', 'GaAsonSifrontRT.npz')
absmat_path = os.path.join(results_path, 'testing', 'GaAsonSifrontA.npz')

fullmat = load_npz(mat_path)
absmat = load_npz(absmat_path)

Rf_1 = fullmat[:, :n_a_in, :]
Tf_1 = fullmat[:, n_a_in:, :]
Af_1 = absmat

Rf_2 = fullmat[:, :n_a_in,:]
Tf_2 = fullmat[:, n_a_in:, :]
Af_2 = absmat

a = np.sum(Rf_2[0].todense(), 0) + np.sum(Tf_2[0].todense(), 0) + np.sum(Af_2[0].todense(), 0)
print(np.all(np.abs(a-1) < 1e-9)) # rounding errors

# rear incidence matrices (back surface)
mat_path = os.path.join(results_path, 'testing', 'GaAsonSirearRT.npz')
absmat_path = os.path.join(results_path, 'testing', 'GaAsonSirearA.npz')

fullmat = load_npz(mat_path)
absmat = load_npz(absmat_path)

Rb_1 = fullmat[:, :n_a_in,:] # first index is wavelength, second is angle out, last is angle in
Tb_1 = fullmat[:, n_a_in:, :]
Ab_1 = absmat

print(np.sum(Rf_1[0].todense(), 0))
print(np.sum(Tf_1[0].todense(), 0))
print(np.sum(Af_1[0].todense(), 0))

a = np.sum(Rf_1[0].todense(), 0) + np.sum(Tf_1[0].todense(), 0) + np.sum(Af_1[0].todense(), 0)
print(np.all(np.abs(a-1) < 1e-9)) # rounding errors

# v0.groupby('wl').apply(lambda x: dot(R, x.data))

def dot_wl(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1]))
    for i1 in range(vec.shape[0]):
        result[i1, :] = dot(mat[i1], vec[i1])
    return result

def dot_wl_prof(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1], mat.shape[3]))
    for i1 in range(vec.shape[0]):
        result[i1, :, :] = dot(mat[i1], vec[i1])
    return result

n_layers = 1

a_1 = []
vr_1 = []
a_2 = []
vt_2 = []
A_1 = []

vf_1 = dot_wl(Tf_1, v0) # pass through front surface
vr_1.append(dot_wl(Rf_1, v0)) # reflected from front surface
a_1.append(dot_wl(Af_1, v0)) # absorbed in front surface at first interaction
power = np.sum(vf_1, axis=1)
remaining_power = [np.array([1]*num_wl)]
# rep
i1=1

while np.any(power > 1e-15):
    print(i1)
    # vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
    # vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    # vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
    # vf_3 = dot_wl(Rb_1, vf_2) # reflect from front surface
    # power = np.sum(vf_3, axis=1)
    vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
    remaining_power.append(np.sum(vb_1, axis=1))
    A_1.append(np.sum(vf_1, 1) - np.sum(vb_1, 1))
    vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
    remaining_power.append(np.sum(vf_2, axis=1))
    A_1.append(np.sum(vb_2, 1) - np.sum(vf_2, 1))
    vf_1 = dot_wl(Rb_1, vf_2) # reflect from front surface
    power = np.sum(vf_1, axis=1)

    vr_1.append(dot_wl(Tb_1, vf_2))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
    vt_2.append(dot_wl(Tf_2, vb_1))  # transmitted into medium below through back surface
    a_2.append(dot_wl(Af_2, vb_1))  # absorbed in 2nd surface
    a_1.append(dot_wl(Ab_1, vf_2))  # absorbed in 1st surface (from the back)

    i1+=1

a_1 = np.array(a_1)
vr_1 = np.array(vr_1)
a_2 = np.array(a_2)
vt_2 = np.array(vt_2)
A_1 = np.array(A_1)
remaining_power = np.array(remaining_power)

# other results:

# end rep
R = np.sum(vr_1, (0,2))
T = np.sum(vt_2, (0,2))
A = np.sum(A_1, 0)
A_front = np.sum(a_1, (0,2))
A_back = np.sum(a_2, (0,2))
plt.figure()

plt.plot(wls, R)
plt.plot(wls, A)
plt.plot(wls, A_front)
plt.plot(wls, A_back)
plt.plot(wls, T)
plt.plot(wls, R+A+A_front+A_back+T)
plt.legend(['R', 'A', 'Af', 'Ab', 'T'])

#a_l_front = np.sum(a_1, 0)
#a_l_back = np.sum(a_2, 0)
#plt.figure()
#plt.plot(wls, a_l_front[:,0])
#plt.plot(wls, a_l_front[:,1])
#plt.legend(np.arange(2))

pr = load_npz('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsonSifrontAprof.npz')

lookuptable = xr.open_dataset('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsonSi.nc')

thetas = np.unique(thetas)
pr = xr.DataArray(pr.todense(), dims = ['wl', 'local_theta', 'global_index'],
                  coords = {'wl': wls*1e9, 'local_theta': thetas,
                            'global_index': np.arange(0, len(angle_vector)/2)})

side = 1
pol = 's'

data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=pr.coords['local_theta'],
                                                     wl=pr.coords['wl'], method='nearest')

def scale_func(x):
    return x.data[:,:, None, None]*scale_params

def select_func(x):
    return (x.data[:,:,None, None] != 0)*const_params

params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('wl', 'local_theta', 'layer', 'coeff')

scale_params = params.loc[dict(coeff=['A1', 'A2', 'A3_r', 'A3_i'])] # have to scale these to make sure integrated absorption is correct
const_params = params.loc[dict(coeff=['a1', 'a3'])] # these should not be scaled

scale_res = pr.groupby('global_index').apply(scale_func)
const_res = pr.groupby('global_index').apply(select_func)

params = xr.concat((scale_res, const_res), dim='coeff')

z = xr.DataArray(np.arange(0, depth, 0.5))

def profile_per_layer(x):
    non_zero = x[np.all(x, axis=1)]
    A1 = non_zero.loc[dict(coeff='A1')]
    A2 = non_zero.loc[dict(coeff='A2')]
    A3_r = non_zero.loc[dict(coeff='A3_r')]
    A3_i = non_zero.loc[dict(coeff='A3_i')]
    a1 = non_zero.loc[dict(coeff='a1')]
    a3 = non_zero.loc[dict(coeff='a3')]
    print(A1.shape, a1.shape, z.shape)
    part1 = A1* np.exp(a1 * z)
    part2 = A2* np.exp(-a1 * z)
    part3 = (A3_r + 1j*A3_i)* np.exp(1j * a3 * z)
    part4 = (A3_r - 1j*A3_i) * np.exp(-1j * a3 * z)
    result = np.real(part1 + part2 + part3 + part4)

    #print(result)
    return result.reduce(np.sum, axis=0)

def profile_per_angle(x):
    # print(x)
    by_layer = x.groupby('layer').apply(profile_per_layer)
    # A1 = x.loc[dict(coeff='A1')]
    # A2 = x.loc[dict(coeff='A2')]
    # A3 = x.loc[dict(coeff='A3')]
    # a1 = x.loc[dict(coeff='a1')]
    # a3 = x.loc[dict(coeff='a3')]
    # part1 = A1* np.exp(a1 * z)
    # part2 = A2* np.exp(a1 * z)
    # part3 = A3* np.exp(1j * a3 * z)
    # part4 = np.conj(A3) * np.exp(-1j * a3 * z)
    # result = part1 + part2 + part3 + part4
    #print(result)
    return by_layer

def scaled_profile(x):
    print('wl')
    by_angle = x.groupby('global_index').apply(profile_per_angle)
    return by_angle

start = time()
ans = params.groupby('wl').apply(scaled_profile)
ans = ans.fillna(0)
# nans in ans; should be zero
print('Took ' + str(time()-start) + ' seconds')

plt.figure()
#plt.plot(z.data+250, ans[0,0,0])
plt.plot(z.data, ans[0,0,0])
plt.plot(z.data, ans[0,100,0])
plt.plot(z.data, ans[0,400,0])
plt.plot(z.data, ans[0,700,0])
plt.plot(z.data, ans[0,1200,0])
plt.legend([angle_vector[0],angle_vector[100], angle_vector[400], angle_vector[700], angle_vector[1200]])
plt.show()

A1 = params.loc[dict(coeff='A1')]
A2 = params.loc[dict(coeff='A2')]
A3_r = params.loc[dict(coeff='A3_r')]
A3_i = params.loc[dict(coeff='A3_i')]
a1 = params.loc[dict(coeff='a1')]
a3 = params.loc[dict(coeff='a3')]


int = ((A1 / a1) * (np.exp(a1 * depth) - 1) - (A2 / a1) * (np.exp(-a1 * depth) - 1) - \
        1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * depth) - 1) + 1j * ((A3_r - 1j * A3_i) / a3) * (
                    np.exp(-1j * a3 * depth) - 1)).fillna(0)

int = int.reduce(np.sum, 'local_theta')

int = int.transpose('wl', 'layer', 'global_index')

#scale = (Ab_1.todense()/int).fillna(0)

#ans = scale*ans
int = int[:,0,:]

if side == -1:
    profile = np.flip(ans, 3)
else:
    profile = ans

profile = profile[:,:, 0,:]
#plt.figure()
#plt.plot(scale[0,0].data)

#plt.show()

plt.figure()
plt.plot(z,profile[0,700])
plt.plot(z,ans[0,700,0])
plt.show()


#vectorxr = xr.DataArray(vb_1, dims = ['wl', 'global_index'], coords = {'wl': profile.coords['wl'], 'global_index': np.arange(0, len(angle_vector)/2)})

#tr = xr.dot(v0, profile, dims = 'global_index')



# manual
#tr_man = np.zeros((1300, 500))
#for i1 in range(len(vectorxr[0])):
#    res = profile[0, i1, :]*vectorxr[0, i1]
#    tr_man[i1,:] = res

#tr_man = np.sum(tr_man, 0)
# integration

#print(np.trapz(profile[0,0,700], x=z))
#print(Af_1[0,0,700])



pr = load_npz('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsonSirearAprof.npz')

thetas = np.unique(thetas)
pr = xr.DataArray(pr.todense(), dims = ['wl', 'local_theta', 'global_index'],
                  coords = {'wl': wls*1e9, 'local_theta': thetas, 'global_index': np.arange(0, len(angle_vector)/2)})

side = -1
pol = 's'

data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=pr.coords['local_theta'],
                                                     wl=pr.coords['wl'], method='nearest')

params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('wl', 'local_theta', 'layer', 'coeff')

scale_params = params.loc[dict(coeff=['A1', 'A2', 'A3_r', 'A3_i'])] # have to scale these to make sure integrated absorption is correct
const_params = params.loc[dict(coeff=['a1', 'a3'])] # these should not be scaled

scale_res = pr.groupby('global_index').apply(scale_func)
const_res = pr.groupby('global_index').apply(select_func)

params = xr.concat((scale_res, const_res), dim='coeff')


start = time()
ans = params.groupby('wl').apply(scaled_profile)
ans = ans.fillna(0)
# nans in ans; should be zero
print('Took ' + str(time()-start) + ' seconds')


A1 = params.loc[dict(coeff='A1')]
A2 = params.loc[dict(coeff='A2')]
A3_r = params.loc[dict(coeff='A3_r')]
A3_i = params.loc[dict(coeff='A3_i')]
a1 = params.loc[dict(coeff='a1')]
a3 = params.loc[dict(coeff='a3')]


int_back = ((A1 / a1) * (np.exp(a1 * depth) - 1) - (A2 / a1) * (np.exp(-a1 * depth) - 1) - \
        1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * depth) - 1) + 1j * ((A3_r - 1j * A3_i) / a3) * (
                    np.exp(-1j * a3 * depth) - 1)).fillna(0)

int_back = int_back.reduce(np.sum, 'local_theta')

int_back = int_back.transpose('wl', 'layer', 'global_index')

int_back = int_back[:,0,:]
#scale = (Ab_1.todense()/int).fillna(0)

#ans = scale*ans

if side == -1:
    profile_back = np.flip(ans, 3)
else:
    profile_back = ans

profile_back = profile_back[:,:, 0,:]

#plt.figure()
#plt.plot(scale[0,0].data)

#plt.show()

plt.figure()
plt.plot(z,profile_back[0,700])
plt.plot(z,ans[0,700,0])
plt.show()


# integration

#print(np.trapz(profile_back[0,0,0], x=z))
#print(Ab_1[0,0,0])

# calculating overall profile:
# need to scale by total incident power at each interaction; profile is assuming total incident power is 1

bulkA = A_1[:,0]
N = len(bulkA)
bulkA = np.vstack((np.arange(0,N), bulkA))
frontA = a_1[:,0,0]
frontA =  np.vstack((np.insert(np.arange(1, N, 2),0,0), frontA))
backA = a_2[:,0,0]
backA =  np.vstack((np.arange(0, N, 2), backA))
R = np.sum(vr_1[:,0], 1)
R =  np.vstack((np.insert(np.arange(1, N, 2),0,0), R))
T = np.sum(vt_2[:,0], 1)
T =  np.vstack((np.arange(0, N, 2), T))
rem_power = np.vstack((np.arange(0,N), remaining_power[1:,0]))


plt.figure()
plt.plot(bulkA[0], bulkA[1], '-o')
plt.plot(frontA[0], frontA[1], '-o')
plt.plot(backA[0], backA[1], '-o')
plt.plot(R[0], R[1], '-o')
plt.plot(T[0], T[1], '-o')
plt.plot(rem_power[0], rem_power[1], '--')
plt.xlim(0,5)
plt.legend(['bulk', 'front', 'back', 'R', 'T', 'remaining power'])
plt.show()



a_1 = []
vr_1 = []
a_2 = []
vt_2 = []
A_1 = []
a_1prof = []
a_2prof = []

vf_1 = dot_wl(Tf_1, v0) # pass through front surface
vr_1.append(dot_wl(Rf_1, v0)) # reflected from front surface
a_1.append(dot_wl(Af_1, v0)) # absorbed in front surface at first interaction
v_xr = xr.DataArray(v0, dims = ['wl', 'global_index'],
                                   coords = {'wl': profile.coords['wl'],
                                             'global_index': np.arange(0, len(angle_vector)/2)})
int_power = xr.dot(v_xr, int, dims = 'global_index')
scale = (np.sum(dot_wl(Af_1,v0), 1)/int_power).fillna(0)
a_1prof.append((scale*xr.dot(v_xr, profile, dims = 'global_index')).data)
power = np.sum(vf_1, axis=1)

# rep
i1=1

while np.any(power > 1e-15):
    print(i1)
    # vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
    # vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    # vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
    # vf_3 = dot_wl(Rb_1, vf_2) # reflect from front surface
    # power = np.sum(vf_3, axis=1)
    vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards

    v_xr = xr.DataArray(vb_1, dims=['wl', 'global_index'],
                        coords={'wl': profile.coords['wl'],
                                'global_index': np.arange(0, len(angle_vector) / 2)})
    int_power = xr.dot(v_xr, int, dims='global_index')
    scale = (np.sum(dot_wl(Af_2, vb_1), 1) / int_power).fillna(0)
    a_2prof.append((scale * xr.dot(v_xr, profile, dims='global_index')).data)


    #remaining_power.append(np.sum(vb_1, axis=1))
    A_1.append(np.sum(vf_1, 1) - np.sum(vb_1, 1))
    vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards

    v_xr = xr.DataArray(vf_2, dims=['wl', 'global_index'],
                        coords={'wl': profile.coords['wl'],
                                'global_index': np.arange(0, len(angle_vector) / 2)})
    int_power = xr.dot(v_xr, int_back, dims='global_index')
    scale = (np.sum(dot_wl(Ab_1, vf_2), 1) / int_power).fillna(0)
    a_1prof.append((scale * xr.dot(v_xr, profile_back, dims='global_index')).data)

    #remaining_power.append(np.sum(vf_2, axis=1))
    A_1.append(np.sum(vb_2, 1) - np.sum(vf_2, 1))
    vf_1 = dot_wl(Rb_1, vf_2) # reflect from front surface
    power = np.sum(vf_1, axis=1)

    vr_1.append(dot_wl(Tb_1, vf_2))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
    vt_2.append(dot_wl(Tf_2, vb_1))  # transmitted into medium below through back surface
    a_2.append(dot_wl(Af_2, vb_1))  # absorbed in 2nd surface
    a_1.append(dot_wl(Ab_1, vf_2))  # absorbed in 1st surface (from the back)

    i1+=1

a_1 = np.array(a_1)
vr_1 = np.array(vr_1)
a_2 = np.array(a_2)
vt_2 = np.array(vt_2)
A_1 = np.array(A_1)

a_1prof = np.array(a_1prof)
a_2prof = np.array(a_2prof)

a_1profsum = np.sum(a_1prof, 0)
a_2profsum = np.sum(a_2prof, 0)
#
# plt.figure()
# plt.plot(z, a_1prof[:4,0,:].transpose())
# plt.show()
#
# plt.figure()
# plt.plot(z, np.log(a_2prof[:4,0,:].transpose()))
# plt.show()
#
# plt.figure()
# plt.plot(z, a_2profsum.transpose())
# plt.show()
# # other results:
#
# # end rep
# R = np.sum(vr_1, (0,2))
# T = np.sum(vt_2, (0,2))
# A = np.sum(A_1, 0)
# A_front = np.sum(a_1, (0,2))
# A_back = np.sum(a_2, (0,2))
# plt.figure()
#
# plt.plot(wls, R)
# plt.plot(wls, A)
# plt.plot(wls, A_front)
# plt.plot(wls, A_back)
# plt.plot(wls, T)
# plt.plot(wls, R+A+A_front+A_back+T)
# plt.legend(['R', 'A', 'Af', 'Ab', 'T'])

print(A_front)
print(np.trapz(a_1profsum, z))

print(A_back)
print(np.trapz(a_2profsum, z))