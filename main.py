import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import loadlabel
print(cv2.__version__)

pathL = "C:/Users/YK/Desktop/dataset/left/"  # left Image url
pathR = "C:/Users/YK/Desktop/dataset/right/" # right Image url
#pathSave = "C:/Users/YK/Desktop/dataset/depth+opening5/"
file_list = os.listdir(pathL)
stereo = cv2.StereoBM_create(numDisparities=0, blockSize=15)
lb = loadlabel.load_label()
label_data = lb.load_txt("0000.txt")
print(label_data)
#f= open('dep.txt','a')
while True:
    for i in range(len(file_list)):
        #img = cv2.imread(pathSave+i)
        imgL = cv2.imread(pathL + file_list[i])
        imgL_removeDC = cv2.imread(pathL + file_list[i])

        for j in label_data[i]:
            if j[1] != 'DontCare':
                imgL_removeDC = cv2.rectangle(imgL_removeDC,(int(float(j[5])),int(float(j[6]))) , (int(float(j[7])),int(float(j[8]))), (0,0,255), 3)
                imgL_removeDC = cv2.putText(imgL_removeDC, j[1], (int(float(j[5])),int(float(j[6]))), 0,1,(64,255,64),2,cv2.LINE_AA)
                imgL = cv2.rectangle(imgL, (int(float(j[5])), int(float(j[6]))), (int(float(j[7])), int(float(j[8]))),(0, 0, 255), 3)
                imgL = mgL_removeDC = cv2.putText(imgL, j[1], (int(float(j[5])), int(float(j[6]))), 0, 1, (64, 255, 64),2, cv2.LINE_AA)
            else:
                imgL = cv2.rectangle(imgL,(int(float(j[5])),int(float(j[6]))) , (int(float(j[7])),int(float(j[8]))), (0,0,255), 3)
                imgL = cv2.putText(imgL, j[1], (int(float(j[5])),int(float(j[6]))), 0,1,(64,255,64),2,cv2.LINE_AA)
        cv2.imshow('imgL', imgL)
        cv2.imshow('imgL_removeDontCare', imgL_removeDC)
        #cv2.imshow('result',result)
        #cv2.imshow('img', result)
        key = cv2.waitKey(50)
        if key == 27:
            print('Pressed ESC')
           # f.close()
            break
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
