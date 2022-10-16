import os
import numpy as np

def data_loader(path):
    '''Loads data contained in the provided path.
    Args:
        path: Path to file or folder containing training data.
    Returns:
        images: [N, height, width, 3]. N RGB views of the scene.
        poses: [N, 4, 4]. N camera poses.
        focal: float. Focal lenght of the camera.i
    '''
    if not os.path.exists(path):
        print('ERROR: Training data not found.')
        print('')
        exit()

    data = np.load(path)
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    return images, poses, focal
