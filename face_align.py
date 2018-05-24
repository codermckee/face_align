# coding: utf-8
import cv2
import dlib
import numpy as np



detector = dlib.get_frontal_face_detector()
point_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#得到左右眼角的两个点
def get_src(pic):
    img = cv2.imread(pic)
    img_out = img.copy()
    dets = detector(img, 0)  # dlib人脸检测
    print dets
    shape = point_predictor(img, dets[0])
    print shape
    left_eye_x = shape.part(39).x * 1.0
    left_eye_y = shape.part(39).y * 1.0
    right_eye_x = shape.part(42).x * 1.0
    right_eye_y = shape.part(42).y * 1.0
    middle_x = shape.part(27).x * 1.0
    middle_y = shape.part(27).y * 1.0

    eye = []
    eye.append(left_eye_x)
    eye.append(left_eye_y)
    # eye.append(middle_x)
    # eye.append(middle_y)
    eye.append(right_eye_x)
    eye.append(right_eye_y)
    eye = np.array(eye).reshape(-1, 2)
    # eye.dtype = 'float32'
    eye = np.float32(eye)
    return eye,img_out


#得到仿射变换参数
def get_affine(src,dst,num):
    a = np.zeros((num*2,4))
    b = np.zeros((num*2,1))
    for i in range(num):
        a[i*2,0] = src[i][0]
        a[i*2,1] = -src[i][1]
        a[i*2,2] = 1
        a[i*2,3] = 0
        a[i*2+1,0] = src[i][1]
        a[i*2+1,1] = src[i][0]
        a[i*2+1,2] = 0
        a[i*2+1,3] = 1
        b[i*2,0] = dst[i][0]
        b[i*2+1,0] = dst[i][1]
    x = np.linalg.solve(a,b)
    return x

#eye,img_out = get_src('005981.jpg')
#x = get_affine(eye,dst,2)

def get_param(x):
    param = np.zeros((2,3))
    param[0,0] = x[0]
    param[0,1] = -x[1]
    param[0,2] = x[2]
    param[1,0] = x[1]
    param[1,1] = x[0]
    param[1,2] = x[3]
    return param
#param = get_param(x)
#new_img = cv2.warpAffine(img_out,param,(250,250))
#cv2.imwrite('125.jpg',new_img)
#M = cv2.getAffineTransform(eye,dst)

#new_img = cv2.warpAffine(img_out,M,(200,200))
