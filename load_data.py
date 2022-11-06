import os
import torch
import imageio
import numpy as np
from typing import Tuple, List, Union, Callable

def load_tiny(
    basedir: str
    ) -> Tuple:
    r'''Loads tiny NeRF data.
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
    """Loads blender datased.
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
    imgs = (np.concat(imgs, axis=0) / 255.).astype(np.float32)
    poses = np.concat(poses, axis=0).astype(np.float32)
    
    # Load depth information
    fname_d = os.path.join(basedir, meta['depth_path'] + '.npz') 
    d_data = np.load(fname_d)
    d_masks = d_data['masks']
    d_ivals = d_data['intervals'] 

    # Compute image height, width and camera's focal length
    H, W = imgs.shape[1:3]
    fov_x = meta['camera_angle_x'] # Field of view along camera x-axis
    focal = 0.5 * W / np.tan(0.5 * fov_x)
    hwf = [H, W, focal]
   
    return imgs, poses, hwf, d_masks, d_ivals
