from __future__ import print_function
import argparse
import os
import sys
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.add_dll_directory(py_dll_path)

from utils.utils import *
import pandas as pd
from PIL import Image
from detector import Yolo
import matplotlib

matplotlib.use('TkAgg')

import time
import argparse
from filterpy.kalman import KalmanFilter
import torchvision
import MOT_metrics
from embeddingNet import EmbeddingNet, TripletNet, Net
from RIAMNet import RIAMNet
np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt, det_features, trks_features):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    #print(xx1)
    #print('aaa')
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    #print(o)
    o = o * 0.1
    #print('---')
    for i in range(len(o)):
        for j in range(len(o[i])):
            #temp = [det_features[i],trks_features[j]]
            #print(temp.shape)
            dis = det_features[i][0]-trks_features[j][0]
            #print(torch.norm(dis))
            #print('dis')
            o[i][j] += 0.9*(1-(torch.norm(dis,p=2)))
    #print(o)
    #print('123')
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, frame, embeddingFunc, transform):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        #print(type(bbox),type(bbox[0]))
        self.represent_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] # 이미지 저장
        #cv2.destroyWindow('afd')
        #cv2.imshow('afd', self.represent_image)
        #self.represent_image = np.array(self.represent_image)
        self.represent_image = Image.fromarray(self.represent_image,mode='RGB')
        self.transform = transform
        self.represent_image = self.transform(self.represent_image)
        #print()
        self.emdeddingFunc = embeddingFunc
        if embeddingFunc != None :
            self.feature = embeddingFunc.get_embedding(self.represent_image.unsqueeze(0))
        #print(len(self.feature[0]))
        #print('********')

        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 상태변수 7, measurement input 4
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])  # transfer matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])  # measurement function
        #print('R')
        #print(self.kf.R)
        self.kf.R[2:, 2:] *= 10.  # measurement noise
        #print('R + noise')
        #print(self.kf.R)
        #print('P')
        #print(self.kf.P)
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.  # covariance
        #print('P + noise')
        #print(self.kf.P)
        #print('Q')
        #print(self.kf.Q)
        self.kf.Q[-1, -1] *= 0.01  # process noise
        self.kf.Q[4:, 4:] *= 0.01
        #print('Q+noise')
        #print(self.kf.Q)

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls = bbox[5]
        #self.represent_image = []

    def update(self, bbox, frame):
        """
        Updates the state vector with observed bbox.
        """
        self.represent_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        self.represent_image = Image.fromarray(self.represent_image, mode='RGB')
        self.represent_image = self.transform(self.represent_image)
        self.feature = self.emdeddingFunc.get_embedding(self.represent_image.unsqueeze(0))
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, det_features, trks_features,iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers, det_features, trks_features)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, Embedding = None):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.EmbeddingFunc = Embedding
    self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

  def update(self, dets=np.empty((0, 5)),frame=np.empty((1024,255))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    trks_feature = []
    to_del = []
    ret = []
    cls = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      trks_feature.append(self.trackers[t].feature)
      #frame = cv2.rectangle(frame, (int(pos[0]), int(pos[1])), (int(pos[2]), int(pos[3])), (255, 0, 0), 3)
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    det_features = []
    for d in range(len(dets)):
      represent_image = frame[int(dets[d,1]):int(dets[d,3]), int(dets[d,0]):int(dets[d,2])]
      represent_image = Image.fromarray(represent_image, mode='RGB')
      represent_image = self.transform(represent_image)
      feature = self.EmbeddingFunc.get_embedding(represent_image.unsqueeze(0))
      det_features.append(feature)
    #print(len(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,det_features, trks_feature,self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :],frame)

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:],frame,self.EmbeddingFunc,self.transform)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            cls.append(trk.cls)
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret),cls
    return np.empty((0,5)),np.empty((0,5))

def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None


set_pandas_display_options()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args



total_time = 0.0
total_frames = 0
a = Yolo()
#embedding_model = EmbeddingNet() #googleNet backbone
#modelname ='210324_DS_checkpoint'
modelname = 'RIAM_finetuning_checkpoint'
#embedding_model = RIAMNet()
checkpoint = torch.load('./model/'+modelname)
#embedding_model.load_state_dict(checkpoint['model_state_dict'])
#embedding_model.eval()
#model = TripletNet(embedding_model)
model = RIAMNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(model)

frame = cv2.imread('./data/samples/000002.png')
path = 'H:/Trackingset'
filelist = os.listdir(path)
print(filelist)
file_array = []
for i in range(len(filelist)):
    npath = path+'/'+filelist[i]
    filelist2 = os.listdir(npath)
    file_array.append([])
    for j in filelist2:
        file_array[i].append(npath+'/'+j)

#print(frame.shape)
#GT_path = '/label_02'
video_num =1
for video in file_array[1:-1]:
    mot_tracker = Sort(max_age=3,
                       min_hits=3,
                       iou_threshold=0.3,
                       Embedding=model)  # create instance of the SORT tracker
    #FILE#
    f_out = open(path + '/tracker_result/' +modelname+'_'+filelist[video_num] + '.txt', 'w')
    f = open(path + '/label_02/' + filelist[video_num] + '.txt', 'r')
    video_num+=1
    lines = f.readlines()
    txt_index = 0
    mtr = MOT_metrics.Motmetrics()
    for f in video:
        print(f)
        frame_num = f.split('/')[-1].split('.')[0]
        gt_array = []
        while True:
            try:
                gt_split = lines[txt_index].split(' ')
                if int(frame_num) == int(gt_split[0]):
                    if gt_split[1] != '-1':
                        gt_array.append([float(gt_split[6]),float(gt_split[7]),float(gt_split[8]),float(gt_split[9]),int(gt_split[1])+1])
                    txt_index+=1
                else:
                    break
            except Exception as e:
                #print(e)
                #print(frame_num)
                #print(gt_split)
                break
        #for i in gt_array:
            #print(i)
        frame = cv2.imread(f)
        ori_frame= frame
        result = a.forword(frame)
        #print('*')
        #print(result)
        start_time = time.time()
        try:
            trackers,c = mot_tracker.update(result,ori_frame)
        except Exception as e:
            print(e)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        total_frames += 1
        #print(result)
        if int(frame_num) == 0:
            print(trackers)
            print('@@@@@@@@@@@@')
            print(gt_array)
        try:
            mtr.frame_update(trackers,gt_array)
        except Exception as e:
            print(trackers,gt_array)


        #summary = mtr.mh.compute(mtr.acc, metrics=mtr.mm.metrics.motchallenge_metrics, name='acc')
        #print(summary)

        #print(mtr.get_acc().events)
        #FILE
        for i,d in enumerate(trackers):
            #print(d)
            #d[:4] = list(map(int,d[:4]))
            #print('%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
            #print((d[0], d[1]), (d[2], d[3]))
            f_out.write('%s ' % str(int(frame_num)))
            f_out.write('%s '% str(int(d[4])))
            #print(c[i])
            #print('asdasdsad')
            if c[i] == 0:
                f_out.write('Pedestrian ')
            elif c[i] == 1:
                f_out.write('Truck ')
            elif c[i] == 2:
                f_out.write('Car ')
            elif c[i] == 3:
                f_out.write('Cyclist ')
            elif c[i] == 4:
                f_out.write('Misc ')
            elif c[i] == 5:
                f_out.write('Van ')
            elif c[i] == 6:
                f_out.write('Tram ')
            elif c[i] == 7:
                f_out.write('Person_sitting ')
            f_out.write('-1 -1 -10.0 {0} {1} {2} {3} -1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.000000\n'.format(d[0],d[1],d[2],d[3]))

        #for i, d in enumerate(trackers):
            #frame = cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 3)
            #frame = cv2.putText(frame, str(int(d[4])),(int(d[0]), int(d[1])),1,2,(0, 0, 255), 2)

         #print('-------real value ---------')
        #for i in range(len(result)):
            #result[i] = list(map(int,result[i]))
            #print((result[i][0],result[i][1]), (result[i][2],result[i][3]))
            #frame = cv2.rectangle(frame, (result[i][0],result[i][1]), (result[i][2],result[i][3]),(0,255,255),2)
            #frame = cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 3)

        #cv2.imshow("asd", frame)

        #print(result)
        #cv2.waitKey(0)
    summary = mtr.mh.compute(mtr.acc, metrics=mtr.mm.metrics.motchallenge_metrics, name='acc')
    print(summary)
    f_out.close()
    #break
#if cv2.waitKey(1) == ord('q'):
#    break
