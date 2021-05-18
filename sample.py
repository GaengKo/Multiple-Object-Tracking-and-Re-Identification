import torch
import torchvision
print(torch.__version__)
import cv2
print(cv2.__version__)
import argparse
import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.add_dll_directory(py_dll_path)

from utils.utils import *

import matplotlib

matplotlib.use('TkAgg')

import time
import argparse
from filterpy.kalman import KalmanFilter
import torchvision
import MOT_metrics
from embeddingNet import EmbeddingNet, TripletNet, Net

array = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(type(array))
whereSum = np.where((array > 4), array, np.NaN)
whereSumOdd = np.where((np.sum(array, axis = 1) > 10).reshape(3, -1), np.where( array % 2 == 1, array, 0), 0)

print(whereSum)
print(whereSumOdd)

embedding_model = Net()
checkpoint = torch.load('./model/210324_DS_checkpoint')
model = TripletNet(embedding_model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
from torchsummary import summary as summary_
summary_(embedding_model,(3,224,224),device='cpu')