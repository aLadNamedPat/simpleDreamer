import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutImageDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        r"""
        Args:
            root_dir (str): path to 'rollouts', which contains subfolders run_<timestamp>.
            img_size (int): resize short edge to this and center-crop to img_size×img_size.
        """

        # find all PNG paths under every run_*/frame_*.png
        pattern = os.path.join(root_dir, "run_*", "frame_*.png")
        self.paths = sorted(glob.glob(pattern))
        
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),           # H×W×C [0,255] → C×H×W [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
    


class RolloutLatentDataset(Dataset):
    """
    Each NPZ is expected to contain
        actions   : [T, a_dim]
        latents   : [T, z_dim]            (optional if mu/logvar present)
        mu        : [T, z_dim]            (optional, for re‑sampling)
        logvar    : [T, z_dim] or
        sigma     : [T, z_dim]            (one of logvar / sigma)

    Parameters
    ----------
    root_dir     : directory with run_*/rollout_data.npz
    segment_len  : length of returned sequences (set None to use full rollout)
    sample_latent: if True and (mu,σ) present, draw z ~ N(μ,σ²) every call
    """

    def __init__(self, root_dir, segment_len=128, sample_latent=True):
        super().__init__()
        self.segment_len   = segment_len
        self.sample_latent = sample_latent

        self.files = sorted(glob.glob(os.path.join(root_dir,
                                                   "run_*", "rollout_data.npz")))
        if not self.files:
            raise RuntimeError(f"No NPZ files found in {root_dir}")

        # index = list of (file_idx, start, end) for fast __getitem__
        self.index = []
        for fid, path in enumerate(self.files):
            with np.load(path) as data:
                T = len(data["actions"])
            if segment_len is None or T <= segment_len:
                self.index.append((fid, 0, T))
            else:
                # non‑overlapping tiles: 0…L, L…2L, …
                for s in range(0, T - segment_len + 1, segment_len):
                    self.index.append((fid, s, s + segment_len))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, start, end = self.index[idx]
        path = self.files[fid]
        data = np.load(path)

        # ----- latent sampling ------------------------------------------- #
        if self.sample_latent and "mu" in data and ("logvar" in data or "sigma" in data):
            mu  = data["mu"][start:end]
            if "logvar" in data:
                std = np.exp(0.5 * data["logvar"][start:end])
            else:
                std = data["sigma"][start:end]
            z = mu + np.random.randn(*std.shape) * std
        else:
            z = data["latents"][start:end]

        # ----- actions & targets ----------------------------------------- #
        a = data["actions"][start:end]

        x = z[:-1]       # input latent
        a = a[:-1]       # action aligned with x
        y = z[1:]        # prediction target

        return (torch.from_numpy(x).float(),
                torch.from_numpy(a).float(),
                torch.from_numpy(y).float(),
                torch.tensor(fid, dtype=torch.long))  # episode id
