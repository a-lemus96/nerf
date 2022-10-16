from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def save_origins_and_dirs(poses):
    '''Plot and save optical axis positions and orientations for each camera pose.
    Args:
        poses: [num_poses, 4, 4]. Camera poses.
    Returns:
        None
    '''
    # Compute optical axis orientations and positions
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses]) 
    origins = poses[:, :3, -1]
    # Plot 3D arrows representing position and orientation of the cameras
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(),
                  dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[..., 2].flatten(),
                  length=0.5,
                  normalize=True)
    plt.savefig('out/verify/poses.png')
    plt.close()
    
def get_rays(height: int,
             width: int,
             focal_length: float,
             camera_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Find origin and direction of rays through every pixel and camera origin.
    Args:
        height: Image height.
        width: Image width.
        focal_length: Focal length of the camera.
        camera_pose: [4, 4]. Camera pose matrix.
    Returns:
        origins_world: [height, width, 3]. Coordinates of rays using world coordinates.
        directions_world: [height, width, 3]. Orientations of rays in world coordinates.
    '''
    # Create grid of coordinates
    i, j = torch.meshgrid(torch.arange(width, dtype=torch.float32).to(camera_pose), 
                          torch.arange(height, dtype=torch.float32).to(camera_pose), 
                          indexing='ij')
    i, j = torch.transpose(i, -1, -2), torch.transpose(j, -1, -2)

    # Use pinhole camera model to represent grid in terms of camera coordinate frame
    directions = torch.stack([(i - width * 0.5) / focal_length, 
                               -(j - height * 0.5) / focal_length,
                               -torch.ones_like(i)], dim=-1)
   
    # Apply camera rotation to ray directions
    directions_world = torch.sum(directions[..., None, :] * camera_pose[:3, :3], axis=-1) 
    # Apply camera translation to ray origin
    origins_world = camera_pose[:3, -1].expand(directions_world.shape)

    return origins_world, directions_world

def sample_stratified(rays_origins: torch.Tensor,
                      rays_directions: torch.Tensor,
                      near: float,
                      far: float,
                      n_samples: int,
                      perturb: Optional[bool] = True,
                      inverse_depth: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Sample along rays using stratified sampling approach.
    Args:
        rays_origins: [height, width, 3]. Ray origins.
        rays_directions: [height, width, 3]. Ray orientations.
        near: Near bound for sampling.
        far: Far bound for sampling.
        n_samples: Number of samples.
        perturb: Use random sampling from within each bin. If disabled, use bin delimiters as sample points.
        inversse_depth:
    Returns:
        points: [height, width, n_samples, 3]. World coordinates for all samples along every ray.  
        z_vals: [height, width, n_samples]. Samples expressed as distances along every ray.
    '''
    # Grab samples for parameter t
    t_vals = torch.linspace(0., 1., n_samples, device=rays_origins.device)
    if not inverse_depth:
        # Sample linearly between 'near' and 'far' bounds.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw samples from rays according to perturb parameter
    if perturb:
        mid_values = 0.5 * (z_vals[1:] + z_vals[:-1]) # middle values between adjacent z points
        upper = torch.concat([mid_values, z_vals[-1:]], axis=-1) # append upper z point to mid values
        lower = torch.concat([z_vals[:1], mid_values], axis=-1) # prepend lower z point to mid values
        t_rand = torch.rand([n_samples], device=z_vals.device) # sample N uniform distributed points 
        z_vals  = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_origins.shape[:-1]) + [n_samples])

    # Compute world coordinates for ray samples
    points = rays_origins[..., None, :] + rays_directions[..., None, :] * z_vals[..., :, None]

    return points, z_vals
