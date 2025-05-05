import torchvision
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random

class CIFAR100_Dataset(Dataset):
    def __init__(self, transform=None, device="cpu", mode="train"):
        self.device = device
        if mode=="train":
            data = torchvision.datasets.CIFAR100(download=True,root="./dataset/train",train=True)
        else:
            data = torchvision.datasets.CIFAR100(download=True,root="./dataset/test",train=False)
        self.data = [[np.array(i[0]), int(i[1])] for i in data]
        random.shuffle(self.data)
        self.transform = transform
        print("Data Loading: Total images in mode: ",mode," : " ,len(self.data))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label_idx = self.data[idx][1]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image.to(self.device), torch.tensor(label_idx, dtype=torch.long).to(self.device)
