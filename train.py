import torch
from torch import nn
from utilities import *
from load_data import *

# Use cuda device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tiny nerf data
images, poses, focal = data_loader('data/tiny_nerf/tiny_nerf_data.npz')
print(f'Images shape: {images.shape}')
print(f'Poses shape: {poses.shape}')
print(f'Focal length: {focal}')
print('')

height, width = images.shape[1:3]
near, far = 2., 6.

n_training_imgs = 100
testimg_idx = 87
testimg, testpose = images[testimg_idx], poses[testimg_idx]

# Plot origins and directions for all camera axes
save_origins_and_dirs(poses)

images = torch.from_numpy(images[:n_training_imgs]).to(device)
poses = torch.from_numpy(poses[:n_training_imgs]).to(device)
focal = torch.from_numpy(focal).to(device)
testimg = torch.from_numpy(testimg).to(device) 
testpose = torch.from_numpy(testpose).to(device) 

with torch.no_grad():
    ray_origins, ray_directions = get_rays(height=height,
                                           width=width,
                                           focal_length=focal,
                                           camera_pose=testpose)
 
# Print information for ray aligned with camera's optical axis
print('Ray Origin:')
print(ray_origins.shape)
print(ray_origins[height // 2, width // 2, :])
print('')

print('Ray Direction:')
print(ray_directions.shape)
print(ray_directions[height // 2, width // 2, :])
print('')

# Gather sample points along each ray for testpose 
ray_origin = ray_origins.view([-1, 3])
ray_direction = ray_directions.view([-1, 3])
n_samples = 8
perturb = True
inverse_depth = False

with torch.no_grad():
    points, z_vals = sample_stratified(rays_origins=ray_origin,
                                       rays_directions=ray_direction,
                                       near=near,
                                       far=far, 
                                       n_samples=n_samples,
                                       perturb=perturb,
                                       inverse_depth=inverse_depth)

print('Input points:')
print(points.shape)
print('')
print('Distances along ray:')
print(z_vals.shape)

# Plot sampled points along a particular ray and save file
y_vals = torch.zeros_like(z_vals).cpu()
# Get unperturbed sampled points along each ray
_, z_vals_unperturbed = sample_stratified(rays_origins=ray_origin,
                                          rays_directions=ray_direction,
                                          near=near,
                                          far=far,
                                          n_samples=n_samples,
                                          perturb=False,
                                          inverse_depth=inverse_depth)

# Plot unperturbed samples in blue
plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].numpy(), 'b-o')
# Plot perturbed samples in red
plt.plot(z_vals[0].cpu().numpy(), y_vals[0].numpy(), 'r-o')
plt.ylim([-1, 2])
plt.title('Sratified Sampling (blue) with Perturbation (red)')
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
plt.grid(True)
plt.savefig('out/verify/stratified_sampling.png')
plt.close()
