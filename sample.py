import torch
import torchvision
print(torch.__version__)
import cv2
print(cv2.__version__)
import numpy as np

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