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
    def __init__(self, root_dir, segment_len = None):
        self.files = glob.glob(os.path.join(root_dir, "run_*", "rollout_data.npz"))
        self.segment_len = segment_len

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        z   = data["latents"]       # [T, z_dim]
        a   = data["actions"]       # [T, a_dim]

        x  = z[:-1]
        a  = a[:-1]
        y  = z[1:]

        if self.segment_len and len(x) > self.segment_len:
            start = np.random.randint(0, len(x) - self.segment_len)
            end   = start + self.segment_len
            x, a, y = x[start:end], a[start:end], y[start:end]

        return torch.from_numpy(x).float(), \
               torch.from_numpy(a).float(), \
               torch.from_numpy(y).float()