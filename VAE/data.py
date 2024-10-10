from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from pathlib import Path
from hashlib import md5
from PIL import Image   
import cv2
import os
import torch


PATH_TO_VIDEO="/mnt/750G/data/v2/video_540ss/"
SAVE_PATH="/home/qwest/data_for_ml/ego4d/"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device {DEVICE} is ready")

data_path = Path(PATH_TO_VIDEO)
data_path_list = list(data_path.glob("*.mp4"))

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


def get_number_of_video():
    print(f"{len(data_path_list)} video in folder {PATH_TO_VIDEO}")

def extract_image(frame, data_path_list, path_to_save, number_video):
    for i in range(number_video):
        # load the video using path of video ( my video length is 37 sec )
        video_path = data_path_list[i].__str__() 
        video = cv2.VideoCapture(video_path)

        success = True
        count = 1
        image_id = 1
        while success:
            success , frame = video.read()
            if success == True:
                
                if count%frame == 0:

                    name = str(md5(frame).hexdigest()[:10])+".jpg"
                    image_id += 1
                    # save the image
                    cv2.imwrite(os.path.join(path_to_save , name),frame)
                count += 1
            else:
                break

        print("Total Extracted Frames :",image_id) 

