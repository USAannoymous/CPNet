import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms
import scipy.misc as m
import scipy.ndimage as n

class nyuDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image = m.imread(image_name)
        depth = m.imread(depth_name)

        image = np.array(image, dtype=np.uint8)
        image = image.astype(float) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        depth = depth.astype(float) / 255.0*10
        depth = torch.from_numpy(depth).float()
        depth = depth.unsqueeze(2)
        depth = depth.transpose(0, 1).transpose(0, 2).contiguous()

        sample = {'image': image, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.frame)

def getTestingData():
    transformed_testing = nyuDataset(csv_file='./data/nyuv2_label13_test.csv')
    dataloader_testing = DataLoader(transformed_testing, batch_size=1 ,shuffle=False, num_workers=10, pin_memory=False)

    return dataloader_testing
