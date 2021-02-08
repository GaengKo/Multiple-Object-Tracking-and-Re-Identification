import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import numpy as np
import cv2
import os
import Sort_tracking as St
import MOT_metrics
np.random.seed(0)



path = 'H:/MOT17/train'

video_list = os.listdir(path)
total_time = 0.0
total_frames = 0
for i in range(len(video_list)):
    mot_tracker = St.Sort(max_age=1,
                       min_hits=3,
                       iou_threshold=0.3)
    if i > 0:
        break
    mtr = MOT_metrics.Motmetrics()
    print('@@@@@@@@@@@')
    dir_gt = path + '/' + video_list[i] + '/gt'
    dir_det = path + '/' + video_list[i] + '/det'
    dir_img1 = path + '/' + video_list[i] + '/img1'

    det_f = open(dir_det+'/det.txt','r')
    det_line = det_f.readlines()
    #det_line = sorted(det_line)
    txt_index = 0
    temp = os.listdir(dir_img1)
    video = []
    for i in temp:
        video.append(dir_img1+'/'+i)
    print(video)

    for frame_num in range(len(video)):
        print(frame_num)
        det_array = []
        while True:
            try:
                det_split = det_line[txt_index].split(',')
                print(det_split)
                if frame_num+1 == int(det_split[0]):
                    if float(det_split[6]) >= -0.2:
                        det_array.append([float(det_split[2]),float(det_split[3]),float(det_split[2])+float(det_split[4]),float(det_split[3])+float(det_split[5]),float(det_split[6])])
                    txt_index += 1
                else:
                    print(det_array)
                    break
            except Exception as e:
                break
        frame = cv2.imread(video[frame_num])
        for d in det_array:

            frame = cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 255), 2)
            #frame = cv2.putText(frame, str(d[4]), (int(d[0]), int(d[1])), 1, 2, (0, 255, 0), 2)
        start_time = time.time()
        print(np.asarray(det_array))
        trackers = mot_tracker.update(np.asarray(det_array), frame)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        total_frames += 1
        for d in trackers:
            #print(d)
            #d[:4] = list(map(int,d[:4]))
            #print('%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
            #print((d[0], d[1]), (d[2], d[3]))
            pass
            #frame = cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 3)
            #frame = cv2.putText(frame, str(int(d[4])),(int(d[0]), int(d[1])),1,2,(0, 0, 255), 2)
        cv2.imshow("asd", frame)
        cv2.waitKey(1)
    det_f.close()
    #for frame in
