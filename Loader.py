import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

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
            transforms.ToTensor(),           # H×W×C [0,255] → C×H×W [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)