from pathlib import Path
from torchvision import transforms
from hashlib import md5
from PIL import Image 
import os
import cv2

path_to_data = "/home/qwest/data_for_ml/mario/"
data_path = Path(path_to_data)
image_path_list = os.listdir(data_path)

absolutle_image = []
for p in image_path_list:
    #print(p)
    png = list(Path(path_to_data+'/'+p+'/').glob("*.png"))
    #print(png)
    absolutle_image.append(png)

transform1 = transforms.Compose([transforms.Resize(110),])

from tqdm import tqdm
for i in tqdm(absolutle_image):
    for y in i:
        byt = cv2.cvtColor(cv2.imread(y), cv2.COLOR_BGR2RGB)
        name = md5(byt).hexdigest()[:12]
        byt = transform1(Image.fromarray(byt))
        byt.save(f'/home/qwest/data_for_ml/MARIO/{name}.png')
