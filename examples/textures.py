import numpy as np
from rayflare.textures import xyz_texture, heights_texture
import matplotlib.pyplot as plt

# provide x, y and z coordinates of points in the unit cell to make a V-groove texture

x = np.array([0, 0, 1, 1, 0.5, 0.5])
y = np.array([0, 1, 0, 1, 0, 1])
z = np.array([0, 0, 0, 0, 1, 1])

[front, back] =  xyz_texture(x, y, z)

fig = plt.figure(figsize=(8,4.5))
ax = plt.subplot(121, projection='3d')
ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
                triangles=front.simplices,  linewidth=1, color = (0.8, 0.8, 0.8, 0.8))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.show()

# fig = plt.figure()
ax2 = plt.subplot(122, projection='3d')
#ax.set_aspect('equal')
ax2.view_init(elev=30., azim=60)
ax2.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices,  linewidth=1, color = (0.5, 0.5, 0.5, 0.5))

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.show()

AFM_data = np.loadtxt('data/pyramids.csv', delimiter=',')
# AFM scan data: grid of heights (z coordinates), x and y dimensions are 20 x 20 um


[front, back] =  heights_texture(AFM_data, 20, 20)




fig = plt.figure(figsize=(8,4.5))
ax = plt.subplot(121, projection='3d')
ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
                triangles=front.simplices,  linewidth=1, color = (0.8, 0.8, 0.8, 0.8))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.show()

# fig = plt.figure()
ax2 = plt.subplot(122, projection='3d')
#ax.set_aspect('equal')
ax2.view_init(elev=30., azim=60)
ax2.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices,  linewidth=1, color = (0.5, 0.5, 0.5, 0.5))

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.show()

