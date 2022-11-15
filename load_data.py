import os
import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import json
from utilities import *
from typing import Tuple, List, Union, Callable

# POSES FROM SPHERICAL COORDINATES

# Translate across world's z-axis
trans_t = lambda t: torch.Tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., t],
                                 [0., 0., 0., 1.]]).float() 

# Rotate around world's x-axis
rot_theta = lambda theta: torch.Tensor([[1., 0., 0., 0.],
                                        [0., np.cos(theta), -np.sin(theta), 0.],
                                        [0., np.sin(theta), np.cos(theta), 0.],
                                        [0., 0., 0., 1.]]).float()

# Rotate around world's z-axis
rot_phi = lambda phi: torch.Tensor([[np.cos(phi), -np.sin(phi), 0., 0.],
                                    [np.sin(phi), np.cos(phi), 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]]).float()

def pose_from_spherical(
    radius: float, 
    theta: float,
    phi: float
    ) -> torch.Tensor:
    r"""Computes 4x4 camera pose from 3D location expressed in spherical coords.
    Camera frame points toward object with its y-axis tangent to the virtual
    spherical surface defined by given radius.
    ---------------------------------------------------------------------------- 
    Args:
        radius: float. Sphere radius.
        theta: 0° < float < 90°. Colatitude angle.
        phi: 0° < float < 360°. Azimutal angle.
    Returns:
        pose: [4, 4]. Camera to world transformation."""

    pose = trans_t(radius) 
    pose = rot_theta(theta/180. * np.pi) @ pose
    pose = rot_phi(phi/180. * np.pi) @ pose 
    
    return pose

# DATA LOADING FUNCTIONS

def load_tiny(
    basedir: str
    ) -> Tuple:
    r'''Loads tiny NeRF data.
    ----------------------------------------------------------------------------
    Args:
        basedir: Basepath that contains tiny NeRF files.
    Returns:
        imgs: [N, H, W, 3]. N HxW RGB images.
        poses: [N, 4, 4]. N 4x4 camera poses.
        focal: float. Camera's focal length.
    '''
    if not os.path.exists(basedir):
        print('ERROR: Training data not found.')
        print('')
        exit()

    data = np.load(basedir)
    imgs = data['images']
    poses = data['poses']
    focal = data['focal']

    return imgs, poses, focal


def load_blender(
    basedir: str,
    ) -> Tuple:
    """Loads blender dataset.
    ----------------------------------------------------------------------------
    Args:
        basedir: str. Basepath that contains all files.
    Returns:
        imgs: [N, H, W, 3]. N HxW RGB images.
        poses: [N, 4, 4]. N 4x4 camera poses.
        hwf: [3, ]. Array containing height, width and focal values.
        d_masks: [N, M]. N sets of pixel masks for M depth intervals.
        d_ivals: [M, 2]. M depth intervals corresponding to depth masks."""
 
    # Load JSON file
    with open(os.path.join(basedir, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)

    # Load frames and poses
    imgs = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(basedir, frame['image_path'] + '.png')
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))

    # Convert to numpy arrays
    imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
    poses = np.stack(poses, axis=0).astype(np.float32)
    
    # Load depth information
    fname_d = os.path.join(basedir, meta['depth_path'] + '.npz') 
    d_data = np.load(fname_d)
    d_masks = d_data['masks']
    d_ivals = d_data['intervals'] 

    # Compute image height, width and camera's focal length
    H, W = imgs.shape[1:3]
    fov_x = meta['camera_angle_x'] # Field of view along camera x-axis
    focal = 0.5 * W / np.tan(0.5 * fov_x)
    hwf = np.array([H, W, np.array(focal)])
  
    imgs = torch.Tensor(imgs[..., :-1]) # discard alpha channel
    poses = torch.Tensor(poses)
    hwf = torch.Tensor(hwf)
    d_masks = torch.Tensor(d_masks)
    d_ivals = torch.Tensor(d_ivals)

    return imgs[...], poses, hwf, d_masks, d_ivals

# NERF DATASET

class DatasetNeRF(Dataset):
    r"""NeRF dataset. A NeRF dataset consists of N x H x W ray origins and
    directions relative to the world frame. Here, N is the number of training
    images of size H x W."""
    def __init__(
        self,
        basedir: str,
        n_imgs: int,
        test_idx: int,
        near: int=2.,
        far: int=7.):

        # Initialize attributes
        self.basedir = basedir
        self.n_imgs = n_imgs
        self.test_idx = test_idx
        self.near = near # near and far sampling bounds for each ray
        self.far = far 

        # Load images, camera poses and depth maps
        data = load_blender(basedir)
        imgs, poses, hwf, d_masks, ivals = data
        self.H, self.W, self.focal = hwf
        
        # Validation image
        self.testimg = imgs[test_idx]
        self.testpose = poses[test_idx]

        # Local rays
        self.local_dirs = get_rays(self.H, self.W, self.focal, local_only=True)
        local_dirs = self.local_dirs[None, None, ...].expand(n_imgs, 1,
                                                             int(self.H),
                                                             int(self.W), 3) 

        # Get rays
        self.rays = torch.stack([torch.stack(
            get_rays(self.H, self.W, self.focal, p), 0)
            for p in poses[:n_imgs]], 0)

        # Append RGB supervision and local rays dirs info
        rays_rgb = torch.cat([self.rays,
                              local_dirs,
                              imgs[:n_imgs, None]], 1)

        # Rearrange data and reshape
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, rays_rgb.shape[3], 3])
        
        self.rays_rgb = rays_rgb.type(torch.float32)

    def __len__(self):
        return self.rays_rgb.shape[0] 

    def __getitem__(self, idx):
        ray_info = self.rays_rgb[idx]
        ray_o, ray_d, local_d, target_pix = ray_info 
        
        return ray_o, ray_d, local_d, target_pix 
