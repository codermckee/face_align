import os
import cv2
import numpy as np
from face_align import get_param,get_src,get_affine
path = './train/1x/'
dst = np.float32([[110, 95], [160, 95]])
for a,b,c in os.walk(path):
    for i in c:
        #img = cv2.imread(path+i)
        print path+i
        eye,img_out = get_src(path+i)
        x = get_affine(eye, dst, 2)
        param = get_param(x)
        new_img = cv2.warpAffine(img_out, param, (260, 260))
        cv2.imwrite('./train/1/'+i,new_img)
        os.remove(path+i)

