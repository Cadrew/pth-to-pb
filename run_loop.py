import torch
import os
import time
from RealESRGAN import RealESRGAN
from PIL import Image

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_path = r'C:\\Users\\adrie\\Downloads\\Recording jeux\\Crash Bandicoot\\cropped'

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        print("Running: " + path)
        st = time.time()
        path_to_image = os.path.join(dir_path, path)
        image = Image.open(path_to_image).convert('RGB')
        sr_image = model.predict(image)
        et = time.time()
        print("elapsed time: " + str(et - st))
        sr_image.save('outputs/' + path)

print("Done")


