from torch.utils.data import Dataset, DataLoader, Sampler
import random
from torchvision.transforms import transforms as T
from pathlib import Path
from hashlib import md5
from PIL import Image   
import cv2
import os
import torch

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

print(f"device {DEVICE} is ready")

# First preprocessing of data
transform1 = T.Compose([
    T.Resize(64),
                        T.CenterCrop(64)
                        ])

# Data augmentation and converting to tensors
transform2 = T.Compose([#T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

class Ego4d(Dataset):
    def __init__(self, img_dir, transform1=None, transform2=None,):
    
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
    
class ROBO(Dataset):
    def __init__(self, root_dir, transform1, transform2):

        self.root_dir = root_dir
        self.image_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')])

        self.transform1 = transform1
        self.transform2 = transform2

        self.imgs = []

        for img in self.image_paths:
            img = Image.open(img)
            
            if self.transform1 is not None:
                img = self.transform1(img)
                
            self.imgs.append(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
            img = self.imgs[index]
            
            if self.transform2 is not None:
                img = self.transform2(img)
            
            return img
    
class ResumableRandomSampler(torch.utils.data.Sampler):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1]

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


class StaticSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
