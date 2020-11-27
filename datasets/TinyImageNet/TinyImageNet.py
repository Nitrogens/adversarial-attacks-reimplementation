import os
import torch
import pandas as pd

from torch.utils.data import Dataset

from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms
        self.filename_list = []

        for i in range(200):
            self.filename_list.append("{:03d}".format(i))

        df = pd.read_csv(self.data_dir + "/data/labels.csv")
        self.labels = df.loc[:, 'TrueLabel'].to_numpy()
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, 'data/images', self.filename_list[idx] + '.png'))
        image = self.transforms(image)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return 200
