from pathlib import Path
from torchvision import transforms
from multiprocessing import *
import numpy
from hashlib import md5
from PIL import Image 
import os
import cv2

path_to_video = "/mnt/750G/data/v2/video_540ss/"

data_path = Path(path_to_video)

print(data_path)

image_path_list = list(data_path.glob("*.mp4"))

transform1 = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64)])

def traf(rangei, additional):
    for i in range(rangei):
        video_path = image_path_list[i+additional].__str__() 
        #print(i+additional, additional)
        video = cv2.VideoCapture(video_path)
        success = True
        count = 1
        image_id = 1
        while success:
            success , frame = video.read()
            if success == True:
                if count%5 == 0:

                    name = str(md5(frame).hexdigest()[:10])+".jpg"
                    image_id += 1
                    if transform1 is not None:
                        img = transform1(Image.fromarray(frame))
            
                        img.save(f'/home/qwest/data_for_ml/test/{name}')
                        #cv2.imwrite(os.path.join('/home/qwest/data_for_ml/test/' , name),frame)
                count += 1
            else:
                break

if __name__ == '__main__':
    processes = []
    for i in range(10):
        process = Process(target=traf, args=(20, i*20))
        processes.append(process)
        process.start()

    # Wait jobs done
    for process in processes:
        process.join()