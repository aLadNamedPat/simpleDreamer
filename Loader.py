import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RolloutImageDataset(Dataset):
    """
    Dataset for loading rollout images for VAE training.
    
    Supports two directory structures:
    1. Old structure: root_dir/run_*/frame_*.png
    2. New unified structure: root_dir/run_*/images/frame_*.png
    
    Automatically detects which structure is present.
    """
    
    def __init__(self, root_dir, img_size=64):
        # Try new unified structure first (run_*/images/frame_*.png)
        pattern_new = os.path.join(root_dir, "run_*", "images", "frame_*.png")
        self.paths = sorted(glob.glob(pattern_new))
        
        # If no images found, try old structure (run_*/frame_*.png)
        if len(self.paths) == 0:
            pattern_old = os.path.join(root_dir, "run_*", "frame_*.png")
            self.paths = sorted(glob.glob(pattern_old))
        
        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                f"Tried patterns:\n"
                f"  - {pattern_new}\n"
                f"  - {pattern_old}\n"
                f"Make sure you've collected rollouts first with collect_rollouts()."
            )
        
        print(f"RolloutImageDataset: Found {len(self.paths)} images in {root_dir}")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class RolloutLatentDataset(Dataset):
    """
    Overlapping sequence dataset for RNN training.
    
    Each NPZ is expected to contain:
        actions   : [T, a_dim]
        latents   : [T, z_dim]            (optional if mu/logvar present)
        mu        : [T, z_dim]            (optional, for re-sampling)
        logvar    : [T, z_dim] or
        sigma     : [T, z_dim]            (one of logvar / sigma)
    
    Parameters
    ----------
    root_dir      : directory with run_*/rollout_data.npz
    seq_len       : length of returned sequences
    sample_latent : if True and (mu,σ) present, draw z ~ N(μ,σ²) every call
    stride        : step between sequence starts (1 = full overlap, seq_len = no overlap)
    split         : 'train', 'test', or None (all data)
    test_ratio    : fraction of files to use for test set (default 0.1)
    test_count    : exact number of files for test set (overrides test_ratio if set)
    seed          : random seed for reproducible train/test split
    """
    
    def __init__(self, root_dir, seq_len=32, sample_latent=True, stride=1,
                 split=None, test_ratio=0.1, test_count=None, seed=42):
        super().__init__()
        self.seq_len = seq_len
        self.sample_latent = sample_latent
        self.stride = stride
        
        all_files = sorted(glob.glob(os.path.join(root_dir, "run_*", "rollout_data.npz")))
        if not all_files:
            raise RuntimeError(
                f"No NPZ files found in {root_dir}. "
                f"Make sure you've collected rollouts and encoded them first."
            )
        
        # Filter out files that don't have latents yet (for RNN training)
        valid_files = []
        for path in all_files:
            with np.load(path) as data:
                # Check if latents are present (either directly or via mu/logvar)
                has_latents = 'latents' in data or 'mu' in data
                if has_latents:
                    valid_files.append(path)
        
        if not valid_files:
            raise RuntimeError(
                f"Found {len(all_files)} NPZ files, but none contain latent data. "
                f"Make sure you've run encode_rollouts() after VAE training."
            )
        
        all_files = valid_files
        print(f"Found {len(all_files)} rollouts with latent data")
        
        # Perform train/test split at the file level
        if split is not None:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(all_files))
            
            if test_count is not None:
                n_test = min(test_count, len(all_files))
            else:
                n_test = int(len(all_files) * test_ratio)
            
            test_indices = set(indices[:n_test])
            train_indices = set(indices[n_test:])
            
            if split == 'test':
                self.files = [all_files[i] for i in sorted(test_indices)]
            elif split == 'train':
                self.files = [all_files[i] for i in sorted(train_indices)]
            else:
                raise ValueError(f"split must be 'train', 'test', or None, got '{split}'")
        else:
            self.files = all_files
        
        # Pre-load all data into memory for fast access
        self.data_cache = []
        for path in self.files:
            with np.load(path) as data:
                self.data_cache.append({
                    'mu': np.copy(data['mu']) if 'mu' in data else None,
                    'logvar': np.copy(data['logvar']) if 'logvar' in data else None,
                    'sigma': np.copy(data['sigma']) if 'sigma' in data else None,
                    'latents': np.copy(data['latents']) if 'latents' in data else None,
                    'actions': np.copy(data['actions']),
                })
        
        # Build index: (file_idx, start_idx) for OVERLAPPING sequences
        self.index = []
        for fid, cached in enumerate(self.data_cache):
            T = len(cached['actions'])
            # Need seq_len + 1 timesteps to create seq_len (input, target) pairs
            for start in range(0, T - seq_len, stride):
                self.index.append((fid, start))
        
        split_str = f" ({split})" if split else ""
        print(f"Dataset{split_str}: {len(self.files)} files, {len(self.index)} sequences "
              f"(seq_len={seq_len}, stride={stride})")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        fid, start = self.index[idx]
        cached = self.data_cache[fid]
        end = start + self.seq_len + 1  # +1 because we need x[:-1] and y[1:]
        
        # ----- latent sampling ------------------------------------------- #
        if self.sample_latent and cached['mu'] is not None:
            mu = cached['mu'][start:end]
            if cached['logvar'] is not None:
                std = np.exp(0.5 * cached['logvar'][start:end])
            else:
                std = cached['sigma'][start:end]
            z = mu + np.random.randn(*std.shape) * std
        else:
            z = cached['latents'][start:end]
        
        # ----- actions & targets ----------------------------------------- #
        a = cached['actions'][start:end]
        
        x = z[:-1]    # [seq_len, z_dim] - input latents
        a = a[:-1]    # [seq_len, a_dim] - actions aligned with x
        y = z[1:]     # [seq_len, z_dim] - target (next latent)
        
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(a).float(),
            torch.from_numpy(y).float(),
        )


def get_train_test_loaders(root_dir, seq_len=32, sample_latent=True, stride=1,
                           test_count=100, batch_size=16, num_workers=4, seed=42):
    """
    Convenience function to create train and test dataloaders.
    
    Parameters
    ----------
    root_dir     : directory with run_*/rollout_data.npz
    seq_len      : length of returned sequences
    sample_latent: if True and (mu,σ) present, draw z ~ N(μ,σ²) every call
    stride       : step between sequence starts
    test_count   : number of files to reserve for test set
    batch_size   : batch size for both loaders
    num_workers  : number of workers for data loading
    seed         : random seed for reproducible split
    
    Returns
    -------
    train_loader, test_loader : DataLoader objects
    """
    train_dataset = RolloutLatentDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        sample_latent=sample_latent,
        stride=stride,
        split='train',
        test_count=test_count,
        seed=seed,
    )
    
    test_dataset = RolloutLatentDataset(
        root_dir=root_dir,
        seq_len=seq_len,
        sample_latent=sample_latent,
        stride=stride,
        split='test',
        test_count=test_count,
        seed=seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test set
        drop_last=False,  # Keep all test samples
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader