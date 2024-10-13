from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from pathlib import Path
from hashlib import md5
from PIL import Image   
import cv2
import os
import torch

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device {DEVICE} is ready")

# First preprocessing of data
transform1 = T.Compose([T.Resize(64),
                        T.CenterCrop(64)])

# Data augmentation and converting to tensors
transform2 = T.Compose([T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

class Ego4d(Dataset):
    def __init__(self, img_dir, transform1=None, transform2=None):
    
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.imgs = []
        for img_name in self.img_names:
            img = Image.open(os.path.join(img_dir, img_name))
            
            if self.transform1 is not None:
                img = self.transform1(img)
                
            self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]
        
        if self.transform2 is not None:
            img = self.transform2(img)
        
        return img

    def __len__(self):
        return len(self.imgs)


