import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from Loader import RolloutImageDataset

# 1) Build the dataset with the same transform you already defined
ds = RolloutImageDataset(root_dir="rollouts", img_size=64)

# 2) Grab a small random batch (here: 16 images)
idxs   = torch.randint(0, len(ds), (16,))
batch  = torch.stack([ds[i] for i in idxs])          # shape (B,3,64,64)

# 3) Arrange them in a 4×4 grid for display
grid = make_grid(batch, nrow=4, padding=2)           # (3,H,W)

# 4) Convert to H×W×C and show
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0))                    # put channels last
plt.axis("off")
plt.title("Resized + Center‑cropped 64×64 inputs")
plt.show()



# 1) Build the dataset with the same transform you already defined
ds = RolloutImageDataset(root_dir="rollouts", img_size=96)

# 2) Grab a small random batch (here: 16 images)
idxs   = torch.randint(0, len(ds), (16,))
batch  = torch.stack([ds[i] for i in idxs])          # shape (B,3,64,64)

# 3) Arrange them in a 4×4 grid for display
grid = make_grid(batch, nrow=4, padding=2)           # (3,H,W)

# 4) Convert to H×W×C and show
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0))                    # put channels last
plt.axis("off")
plt.title("Resized + Center‑cropped 64×64 inputs")
plt.show()