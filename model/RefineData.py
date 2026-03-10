import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class RefineData(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.files = os.listdir(image_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")

        image = torch.tensor(image).permute(2,0,1).float() / 255.0

        sample = {
            "image": image,
            "volume_gt": torch.tensor(1.0)  # placeholder
        }

        return sample