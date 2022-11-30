import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models import *
from utilities import *
from load_data import *
import logging
from tqdm import tqdm

# For repeatability
'''seed = 451
torch.manual_seed(seed)
np.random.seed(seed)'''

# Use cuda device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = DSNerfDataset(basedir='data/bunny/',
                        n_imgs=49,
                        test_idx=49,
                        near=1.2,
                        far=7.)

near, far = dataset.near, dataset.far
H, W, focal = int(dataset.H), int(dataset.W), dataset.focal

testimg, testpose = dataset.testimg, dataset.testpose

logger = logging.getLogger()
base_level = logger.level

# TRAINING CLASSES AND FUNCTIONS

def plot_samples(
    z_vals: torch.Tensor, 
    z_hierarch: Optional[torch.Tensor] = None,
    ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    Args:
    Returns:
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')

    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)

    return ax

def crop_center(
    img: torch.Tensor,
    frac: float = 0.5
    ) -> torch.Tensor:
    r"""
    Crop center square from image.
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))

    return img[h_offset:-h_offset, w_offset:-w_offset]

class EarlyStopping:
    r"""
    Early stopping helper class based on fitness criterion.
    """
    def __init__(
        self,
        patience: int = 30,
        margin: float = 1e-4
    ):
        self.best_fitness = 0.0 # PSNR measure
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float('inf') # epochs to wait after fitness
                                                 # stops improving to stop

    def __call__(
        self,
        iter: int,
        fitness: float
    ):
        r"""
        Check if stopping criterion is met.
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience # stop training if patience is exceeded

        return stop

# HYPERPARAMETERS 

# Encoders
d_input = 3             # Number of input dimensions
n_freqs = 10            # Number of encoding functions for samples
log_space = True        # If set, frecuencies scale in log space
use_viewdirs = True     # If set, model view dependen effects
n_freqs_views = 4       # Number of encoding functions for views

# Stratified sampling
n_samples = 64          # Number of spatial samples per ray
perturb = True          # If set, applies noise to sample positions
inverse_depth = False   # If set, sample points linearly in inverse depth

# Model
d_filter = 128          # Dimension of linear layer filters
n_layers = 8            # Number of layers in network bottleneck
skip = [4]              # Layers at which to apply input residual
use_fine_model = False  # If set, creates a fine model
d_filter_fine = 128     # Dimension of linear layer filters of fine network
n_layers_fine = 6       # Number of layers in fine network bottleneck

# Hierarchical sampling
n_samples_hierarchical = 64     # Number of samples per ray
perturb_hierarchical = True    # If set, applies noise to sample positions

# Optimizer
lrate = 5e-4            # Learning rate

# Training 
n_iters = 1e6
batch_size = 2**12          # Number of rays per gradient step
one_image_per_step = False  # One image per gradient step (disables batching)
chunksize = 2**12           # Modify as needed to fit in GPU memory
center_crop = False         # Crop the center of image (one_image_per_)
center_crop_iters = 100      # Stop cropping center after this many epochs
display_rate = 50           # Display test output every X epochs
val_rate = 25               # Evaluation of test image rate

# Early Stopping
warmup_iters = 2500           # Number of iterations during warmup phase
warmup_min_fitness = 14.5   # Min val PSNR to continue training at warmup_iters
n_restarts = 20             # Number of times to restart if training stalls

# Bundle the kwargs for various functions to pass all at once
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}

kwargs_sample_hierarchical = {
    'perturb': perturb_hierarchical
}

kwargs_sample_normal = {
    'n_samples': n_samples,
    'inverse_depth': inverse_depth
}

# MODELS INITIALIZATION

def init_models():
    r"""
    Initialize models, encoders and optimizer for NeRF training
    """
    # Encoders
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # Check if using view directions to initialize encoders
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                             log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter,
                 skip=skip, d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(encoder.d_output, n_layers=n_layers, 
                          d_filter=d_filter, skip=skip, d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None
    
    return model, fine_model, encode, encode_viewdirs 

# TRAINING LOOP

# Early stopping helper
warmup_stopper = EarlyStopping(patience=100)

def train(mu=0.005):
    r"""
    Run NeRF training loop.
    """
    # Create data loader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CustomScheduler(optimizer, n_iters, n_warmup=warmup_iters)

    train_psnrs = []
    val_psnrs = []
    iternums = []
    
    testimg, testpose = dataset.testimg.to(device), dataset.testpose.to(device)

    # Compute number of epochs
    steps_per_epoch = np.ceil(len(dataset)/batch_size)
    epochs = np.ceil(n_iters / steps_per_epoch)

    for i in range(int(epochs)):
        print(f"Epoch {i + 1}")
        model.train()

        for k, batch in enumerate(tqdm(dataloader)): 
            # Compute step
            step = int(i * steps_per_epoch + k)

            # Unpack batch info
            rays_o, rays_d, target_pixs, t_ivals, bkgd = batch
            bkgd = bkgd.type(torch.bool)
            
            # Send data to GPU
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            t_ivals = t_ivals.to(device)
            
            # Exclude background from depth supervision
            #t_ivals[bkgd] = t_ivals[bkgd]*0.
           
            '''if step < center_crop_iters:
                rand_idx = torch.empty_like(bkgd).type(torch.float32).uniform_() > 0.1
                selection = torch.logical_or(~bkgd, rand_idx)
                #print(torch.sum(selection))
                rays_o = rays_o[selection]
                rays_d = rays_d[selection]
                target_pixs = target_pixs[selection]
                t_ivals = t_ivals[selection]
            else:'''
            selection = torch.ones_like(bkgd).type(torch.bool)

            bkgd = bkgd.to(device)

            # Run one iteration of NeRF and get the rendered RGB image
            outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model, t_ivals,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           kwargs_sample_normal=kwargs_sample_normal,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=rays_d.shape[0])

            # Check for numerical issues
            for key, val in outputs.items():
                if torch.isnan(val).any():
                    print(f"! [Numerical Alert] {key} contains NaN.")
                if torch.isinf(val).any():
                    print(f"! [Numerical Alert] {key} contains Inf.")

            # Send RGB training data to GPU
            target_pixs = target_pixs.to(device)

            # Retrieve predictions from model
            rgb_predicted = outputs['rgb_map']
            d_predicted = outputs['depth_map']
            weights = outputs['weights'] + 1e-12
            #print(f'Weights: {weights.shape}')
            z_vals = outputs['z_vals_stratified']

            # Compute RGB loss
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_pixs)

            # Compute PSNR value
            with torch.no_grad():
                psnr = -10. * torch.log10(loss)
                train_psnrs.append(psnr.item())

            # Compute middle points and interval lengths
            mids = (t_ivals[:, 0] + t_ivals[:, 1]) / 2.            
            mids = mids[..., None]
            boxsize = t_ivals[..., 1] - t_ivals[..., 0]
            boxsize = boxsize[..., None]

            # Compute distances between samples
            dists = z_vals[..., 1] - z_vals[..., 0]
            dists = dists[..., None] 
            # Compute KL depth loss 
            d_loss = torch.log(weights)
            #print(f'D loss after log: {d_loss.shape}')
            d_loss = d_loss * torch.exp(-(z_vals - mids)**2 / (2 * boxsize)) * dists
            #print(f'D loss after exp log: {d_loss}')
            d_loss = torch.sum(d_loss, -1)
            # Remove background from gradient calculation
            d_loss = d_loss * ~bkgd[selection]

            
            # Remove nans from depth loss
            #filter_out = torch.logical_and(~torch.isnan(d_loss), ~torch.isinf(d_loss))
            #print(f'Number of Nans/Infs in d_loss: {torch.sum(~filter_out)}')
            #d_loss = d_loss[filter_out]
            #print(d_loss.shape)
            #print(d_loss)
            #exit()

            # Compute total loss
            loss += mu * torch.mean(d_loss)

            # Perform backprop and optimizer steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % val_rate == 0:
                with torch.no_grad():
                    model.eval()

                    rays_o, rays_d = get_rays(H, W, focal, testpose)
                    rays_o = rays_o.reshape([-1, 3])
                    rays_d = rays_d.reshape([-1, 3])
                    t_ivals = dataset.test_ivals.reshape([-1, 2]).to(device)

                    outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model, t_ivals,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           kwargs_sample_normal=kwargs_sample_normal,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize)
                    
                    # Compute middle points for intervals
                    mids = (t_ivals[:, 0] + t_ivals[:, 1]) / 2.            

                    rgb_predicted = outputs['rgb_map']
                    depth_predicted = outputs['depth_map']

                    val_loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
                    val_psnr = -10. * torch.log10(val_loss)

                    val_psnrs.append(val_psnr.item())
                    iternums.append(step)
                    if step % display_rate == 0:
                        logger.setLevel(100)

                        # Plot example outputs
                        fig, ax = plt.subplots(2, 3, figsize=(25, 8),
                                               gridspec_kw={'width_ratios': [1, 1, 3]})
                        ax[0,0].imshow(rgb_predicted.reshape([H, W, 3]).cpu().numpy())
                        ax[0,0].set_title(f'Iteration: {step}')
                        ax[0,1].imshow(testimg.cpu().numpy())
                        ax[0,1].set_title(f'Target')
                        ax[0,2].plot(range(0, step + 1), train_psnrs, 'r')
                        ax[0,2].plot(iternums, val_psnrs, 'b')
                        ax[0,2].set_title('PSNR (train=red, val=blue')
                        ax[1,0].imshow(depth_predicted.reshape([H, W]).cpu().numpy(),
                                     vmin=0., vmax=7.5)
                        ax[1,0].set_title(r'Predicted Depth')
                        ax[1,1].imshow(mids.reshape([H, W]).cpu().numpy(),
                                     vmin=0., vmax=7.5)
                        ax[1,1].set_title('Target')
                        z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
                        z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
                        if 'z_vals_hierarchical' in outputs:
                            z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
                            z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
                        else:
                            z_sample_hierarch = None
                        _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[1,2])
                        ax[1,2].margins(0)
                        plt.savefig(f"out/training/iteration_{step}.png")
                        plt.close(fig)
                        logger.setLevel(base_level)

            # Check PSNR for issues and stop if any are found.
            if step == warmup_iters - 1:
                if val_psnr < warmup_min_fitness:
                    return False, train_psnrs, val_psnrs, 0
            elif step < warmup_iters:
                if warmup_stopper is not None and warmup_stopper(step, val_psnr):
                    return False, train_psnrs, val_psnrs, 1 

        print("Loss:", val_loss.item())
        #scheduler.step()

    return True, train_psnrs, val_psnrs, 2

# Run training session(s)
for k in range(n_restarts):
    print('Training attempt: ', k + 1)
    model, fine_model, encode, encode_viewdirs = init_models()
    success, train_psnrs, val_psnrs, code = train()

    if success and val_psnrs[-1] >= warmup_min_fitness:
        print('Training successful!')
        break
    if not success and code == 0:
        print(f'Val PSNR {val_psnrs[-1]} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
    elif not success and code == 1:
        print(f'Train PSNR flatlined for {warmup_stopper.patience} iters. Stopping...')


# Compute camera poses along video rendering path
render_poses = [pose_from_spherical(3.5, 45., phi)
                for phi in np.linspace(0, 360, 40, endpoint=False)]
render_poses = torch.stack(render_poses, 0)
render_poses = render_poses.to(device)

# Render frames for all rendering poses
frames = render_path(render_poses=render_poses, 
                     near=near,
                     far=far,
                     hwf=[H, W, focal],
                     encode=encode,
                     model=model,
                     kwargs_sample_stratified=kwargs_sample_stratified,
                     n_samples_hierarchical=n_samples_hierarchical,
                     kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                     fine_model=fine_model,
                     encode_viewdirs=encode_viewdirs,
                     chunksize=chunksize)

# Now we put together frames and save result into .mp4 file
render_video(basedir='out/video/',
             frames=frames)
