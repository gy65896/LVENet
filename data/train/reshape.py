import numpy as np
import cv2
import os

def load_images(file):
    im = cv2.imread(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm


files_clear = os.listdir('./syn2')
for i in range(len(files_clear)):
    #clear = cv2.imread('./syn4/' + files_clear[i])
    img   = load_images('./syn2/' + files_clear[i])
    w,h,c = img.shape
    if w>2000 or h>2000:
        IMG   = cv2.resize(img,(1080,720))
    else:
        IMG   = img#cv2.resize(img,(640,480))#img#cv2.resize(img,(1080,720))
    cv2.imwrite('./syn/'+str(i)+'.png',IMG*255)

