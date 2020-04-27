import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def makeDepth():
    pathL = "C:/Users/YK/Desktop/dataset/left/"  # left Image url
    pathR = "C:/Users/YK/Desktop/dataset/right/" # right Image url
    #pathSave = "C:/Users/YK/Desktop/dataset/depth+opening5/" # path to save
    file_list = os.listdir(pathL)
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=15)
    F_result = []
    for i in file_list:
        imgL = cv2.imread(pathL+i,0)
        imgR = cv2.imread(pathR+i,0)
        disparity = stereo.compute(imgL, imgR)
        kernel = np.ones((5,5), np.uint8)
        result = cv2.morphologyEx(disparity,cv2.MORPH_OPEN,kernel)
        F_result.append(result)
        #plt.imshow(result, 'gray')
        #plt.savefig(pathSave+i,dpi=300)

    return F_result







"""for i in file_list:
    print(i)
    imgL = cv2.imread(pathL+i,0)
    imgR = cv2.imread(pathR+i,0)
    disparity = stereo.compute(imgL, imgR)
    kernel = np.ones((5,5), np.uint8)
    result = cv2.morphologyEx(disparity,cv2.MORPH_OPEN,kernel)
    img_denoise = cv2.GaussianBlur(disparity, (9, 9), 2)
    cv2.imshow('disp',d)
    plt.imshow(result, 'gray')
    plt.savefig(pathSave+i,dpi=300)
    cv2.imshow('imgR', imgR)
    cv2.imshow('imgL', imgL)
    plt.show()
    plt.close()

while True:
    file_list2 = os.listdir(pathSave)
    for i in file_list2:
        #img = cv2.imread(pathSave+i)
        imgL = cv2.imread(pathL + i,0)
        imgR = cv2.imread(pathR + i,0)
        disparity = stereo.compute(imgL, imgR)
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
        plt.imshow(result,'gray')
        break
        #cv2.imshow('imgL', imgL-imgR)
        #cv2.imshow('img', result)
        #key = cv2.waitKey(50)
        #if key == 27:
        #    print('Pressed ESC')
        #    break
#imgL = cv2.imread('C:/Users/YK/Desktop/dataset/left/000111.png',0)
#imgR = cv2.imread('C:/Users/YK/Desktop/dataset/right/000111.png',0)
#imgL = cv2.cv2.resize(imgL, dsize=(300, 300), interpolation=cv2.INTER_AREA)
#imgR = cv2.resize(imgR,(255,255))
#stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
#disparity = stereo.compute(imgL,imgR)
#plt.imshow(disparity,'gray')
#cv2.imshow('imgR',imgR)
#cv2.imshow('imgL',imgL)
#plt.show()
"""