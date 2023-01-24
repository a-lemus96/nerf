# Standard library imports
import argparse
from datetime import date
import logging
import os

# Related third party imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local application/library specific imports
from models import *
from utilities import *
from load_data import *

# For repeatability
'''seed = 451
torch.manual_seed(seed)
np.random.seed(seed)'''

# HYPERPARAMETERS 
parser = argparse.ArgumentParser(description='Train DSNeRF for view synthesis.')

# Encoder(s)
parser.add_argument('--d_input', dest='d_input', default=3, type=int,
                    help='Spatial input dimension')
parser.add_argument('--n_freqs', dest='n_freqs', default=10, type=int,
                    help='Number of encoding functions for spatial coords')
parser.add_argument('--log_space', dest='log_space', default=True, type=bool,
                    help='If set, frecuency scale in log space')
parser.add_argument('--use_viewdirs', dest='use_viewdirs', default=True,
                    type=bool, help='If set, model view dependent effects')
parser.add_argument('--n_freqs_views', dest='n_freqs_views', default=4,
                    type=int, help='Number of encoding functions for view dirs')

# Model(s)
parser.add_argument('--d_filter', dest='d_filter', default=256, type=int,
                    help='Linear layer filter dimension')
parser.add_argument('--n_layers', dest='n_layers', default=8, type=int,
                    help='Number of layers preceding bottleneck')
parser.add_argument('--skip', dest='skip', default=[4], type=list,
                    help='Layers at which to apply input residual')
parser.add_argument('--use_fine', dest='use_fine', default=False, type=bool,
                    help='Creates and uses fine NeRF model')
parser.add_argument('--d_filter_fine', dest='d_filter_fine', default=128,
                    type=int, help='Linear layer filter dim for fine model')
parser.add_argument('--n_layers_fine', dest='n_layers_fine', default=8,
                    type=int, help='Number of fine layers preceding bottleneck')

# Stratified sampling
parser.add_argument('--n_samples', dest='n_samples', default=64, type=int,
                    help='Number of stratified samples per ray')
parser.add_argument('--perturb', dest='perturb', default=True, type=bool,
                    help='Apply noise to spatial coords')
parser.add_argument('--inv_depth', dest='inv_depth', default=False, type=bool,
                    help='Sample points linearly in inverse depth')

# Normal sampling
parser.add_argument('--n_samples_norm', dest='n_samples_norm', default=64,
                    type=int, help='Number of samples from normal distribution')

# Hierarchical sampling
parser.add_argument('--n_samples_hierch', dest='n_samples_hierch', default=64,
                    type=int, help='Number of hierarchical samples per ray')
parser.add_argument('--perturb_hierch', dest='perturb_hierch', default=True,
                    type=bool, help='Applies noise to hierarchical samples')

# Optimization
parser.add_argument('--lrate', dest='lrate', default=5e-4, type=float,
                    help='Learning rate')
parser.add_argument('--mu', dest='mu', default=1e-1, type=float,
                    help='Balancing factor for KL depth loss term')
parser.add_argument('--mu_1', dest='mu_1', default=1e-1, type=float,
                    help='Balancing factor for MSE depth loss term')

# Training 
parser.add_argument('--n_iters', dest='n_iters', default=1e4, type=int,
                    help='Number of training iterations')
parser.add_argument('--batch_size', dest='batch_size', default=2**12, type=int,
                    help='Number of rays per optimization step')
parser.add_argument('--chunksize', dest='chunksize', default=2**12, type=int,
                    help='Batch is divided into chunks to avoid OOM error')

# Validation
parser.add_argument('--display_rate', dest='display_rate', default=50, type=int,
                    help='Display rate for test output measured in iterations')
parser.add_argument('--val_rate', dest='val_rate', default=25, type=int,
                    help='Test image evaluation rate')

# Early Stopping
parser.add_argument('--warmup_iters', dest='warmup_iters', default=100,
                    type=int, help='Number of iterations for warmup phase')
parser.add_argument('--min_fitness', dest='min_fitness', default=14.5,
                    type=float, help='Minimum PSNR value to continue training')
parser.add_argument('--n_restarts', dest='n_restarts', default=10, type=int,
                    help='Maximum number of restarts if training stalls')

args = parser.parse_args()

# Bundle the kwargs for various functions to pass all at once
kwargs_sample_stratified = {
    'n_samples': args.n_samples,
    'perturb': args.perturb,
    'inverse_depth': args.inv_depth
}

kwargs_sample_hierarchical = {
    'perturb': args.perturb_hierch
}

kwargs_sample_normal = {
    'n_samples': args.n_samples_norm,
    'inverse_depth': args.inv_depth
}

# Verify cuda availability
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Device: CPU")

# Use cuda device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set output directories
out_dir = os.path.normpath(os.path.join('out', 'ds_nerf', str(date.today()),
                                        'lrate_' + str(args.lrate),
                                        ('m_' + str(args.mu)
                                         + '_m1_' + str(args.mu_1))))

# Create directories
try:
    os.makedirs(os.path.join(out_dir, 'training'))
except:
    pass

try:
    os.makedirs(os.path.join(out_dir, 'video'))
except:
    pass

try:
    os.makedirs(os.path.join(out_dir, 'model'))
except:
    pass

# Load dataset
dataset = DSNerfDataset(basedir='data/bunny/',
                        n_imgs=5,
                        test_idx=49,
                        f_forward=True,
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


# MODELS INITIALIZATION

def init_models():
    r"""
    Initialize models, encoders and optimizer for NeRF training
    """
    # Encoders
    encoder = PositionalEncoder(args.d_input, args.n_freqs,
                                log_space=args.log_space)
    encode = lambda x: encoder(x)

    # Check if using view directions to initialize encoders
    if args.use_viewdirs:
        encoder_viewdirs = PositionalEncoder(args.d_input, args.n_freqs_views,
                                             log_space=args.log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=args.n_layers, 
                 d_filter=args.d_filter, skip=args.skip,
                 d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if args.use_fine:
        fine_model = NeRF(encoder.d_output, n_layers=args.n_layers, 
                          d_filter=args.d_filter, skip=args.skip,
                          d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None
    
    return model, fine_model, encode, encode_viewdirs 

# TRAINING LOOP

# Early stopping helper
warmup_stopper = EarlyStopping(patience=100)

def train():
    r"""
    Run NeRF training loop.
    """
    # Create data loader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    '''scheduler = CustomScheduler(optimizer,
                                args.n_iters,
                                n_warmup=args.warmup_iters)'''

    train_psnrs = []
    val_psnrs = []
    iternums = []
    
    testimg, testpose = dataset.testimg.to(device), dataset.testpose.to(device)

    # Compute number of epochs
    steps_per_epoch = np.ceil(len(dataset)/args.batch_size)
    epochs = np.ceil(args.n_iters / steps_per_epoch)

    # Initial factor
    #factor = 6.
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

            # Compute middle points and interval lengths
            mids = (t_ivals[:, 0] + t_ivals[:, 1]) / 2.            
            boxsize = t_ivals[..., 1] - t_ivals[..., 0]
            #factor = 0.98 * factor if step < args.warmup_iters else 1.

            # Run one iteration of NeRF and get the rendered RGB image
            '''outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model, mids, boxsize,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           kwargs_sample_normal=kwargs_sample_normal,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=args.chunksize)'''
            outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=args.n_samples_hierch,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=args.chunksize)

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
            z_vals = outputs['z_vals_combined']

            # Compute RGB loss
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_pixs)

            # Compute PSNR value
            with torch.no_grad():
                psnr = -10. * torch.log10(loss)
                train_psnrs.append(psnr.item())

            mids = mids[..., None]
            boxsize = boxsize[..., None]

            # Compute distances between samples
            dists = z_vals[..., 1] - z_vals[..., 0]
            dists = dists[..., None] 

            # Compute KL depth loss 
            d_loss = -torch.log(weights)
            prior = torch.exp(-(z_vals - mids)**2 / (2 * boxsize))
            prior = prior * ~bkgd[:, None] # Set bkgd prior to 0
            d_loss = d_loss * prior * dists
            d_loss = torch.sum(d_loss, -1)
            # Remove background from gradient calculation
            #d_loss = d_loss * ~bkgd[selection]
            
            # Remove nans from depth loss
            #filter_out = torch.logical_and(~torch.isnan(d_loss), ~torch.isinf(d_loss))
            #print(f'Number of Nans/Infs in d_loss: {torch.sum(~filter_out)}')
            #d_loss = d_loss[filter_out]
            #print(d_loss.shape)
            #print(d_loss)
            #exit()

            # Compute total loss
            loss += args.mu * torch.mean(d_loss)
            loss += args.mu_1 * torch.nn.functional.mse_loss(d_predicted,
                                                             mids[:, 0]) 

            # Perform backprop and optimizer steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_lrate = args.lrate * (decay_rate ** (step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if step % args.val_rate == 0:
                with torch.no_grad():
                    model.eval()

                    rays_o, rays_d = get_rays(H, W, focal, testpose)
                    rays_o = rays_o.reshape([-1, 3])
                    rays_d = rays_d.reshape([-1, 3])
                    t_ivals = dataset.test_t_ivals.reshape([-1, 2]).to(device)
                    
                    # Compute middle points and interval lengths
                    mids = (t_ivals[:, 0] + t_ivals[:, 1]) / 2.            
                    #mids = mids[..., None]
                    boxsize = t_ivals[..., 1] - t_ivals[..., 0]
                    #boxsize = boxsize[..., None]

                    '''outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model, mids, boxsize,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           kwargs_sample_normal=kwargs_sample_normal,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=args.chunksize)'''

                    outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=args.n_samples_hierch,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=args.chunksize)

                    mids = mids[..., None]
                    boxsize = boxsize[..., None]

                    rgb_predicted = outputs['rgb_map']
                    depth_predicted = outputs['depth_map']
                    weights = outputs['weights']
                    z_vals = outputs['z_vals_combined']

                    val_loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
                    val_psnr = -10. * torch.log10(val_loss)

                    val_psnrs.append(val_psnr.item())
                    iternums.append(step)
                    if step % args.display_rate == 0:
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
                        # Plot termination distribution and Gaussian distribution
                        n_total= args.n_samples + args.n_samples_hierch
                        t = torch.linspace(1.2, 7.0, n_total, device=device)
                        prior = torch.exp(-(t - mids)**2 / (2 * boxsize))
                        prior = prior * 1/(2 * np.pi * torch.sqrt(boxsize))  
                        ax[1,2].plot(z_vals[64584].cpu().numpy(),
                                weights[64584].detach().cpu().numpy(), 'g-o')
                        ax[1,2].plot(t.cpu().numpy(),
                                prior[64584].detach().cpu().numpy(), 'b')
                        ax[1,2].set_title('Empirical (green) and prior (blue) termination distribution')
                        ax[1,2].margins(0)
                        plt.savefig(f"{out_dir}/training/iteration_{step}.png")
                        plt.close(fig)
                        logger.setLevel(base_level)
                        '''z_vals_strat = outputs['z_vals_stratified'].view((-1, args.n_samples))
                        z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
                        if 'z_vals_hierarchical' in outputs:
                            z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
                            z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
                        else:
                            z_sample_hierarch = None
                        _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[1,2])
                        ax[1,2].margins(0)
                        '''
            # Check PSNR for issues and stop if any are found.
            if step == args.warmup_iters - 1:
                if val_psnr < args.min_fitness:
                    return False, train_psnrs, val_psnrs, 0
            elif step < args.warmup_iters:
                if warmup_stopper is not None and warmup_stopper(step, val_psnr):
                    return False, train_psnrs, val_psnrs, 1 

        print("Loss:", val_loss.item())
        #scheduler.step()

    return True, train_psnrs, val_psnrs, 2

# Run training session(s)
for k in range(args.n_restarts):
    print('Training attempt: ', k + 1)
    model, fine_model, encode, encode_viewdirs = init_models()
    success, train_psnrs, val_psnrs, code = train()

    if success and val_psnrs[-1] >= args.min_fitness:
        print('Training successful!')
        break
    if not success and code == 0:
        print(f'Val PSNR {val_psnrs[-1]} below min_fitness {args.min_fitness}. Stopping...')
    elif not success and code == 1:
        print(f'Train PSNR flatlined for {warmup_stopper.patience} iters. Stopping...')

# Save model
torch.save(model, out_dir + '/model/dsnerf')

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
                     n_samples_hierarchical=args.n_samples_hierch,
                     kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                     fine_model=fine_model,
                     encode_viewdirs=encode_viewdirs,
                     chunksize=args.chunksize)

# Now we put together frames and save result into .mp4 file
render_video(basedir=f'{out_dir}/video/',
             frames=frames)
