import motmetrics as mm
import numpy as np

class Motmetrics():
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)
    def frame_update(self, detector, GT):
        d = []
        Gt = []
        between_iou = []
        for i in GT:
            Gt.append(GT[4])
            for j in range(len(detector)):
                temp = []
                dis = self.iou_batch(j[:4],i[:4])
                temp.append(dis)
            between_iou.append(temp)

        for i in detector:
            d.append(i[4])
        self.acc.update(Gt,d,between_iou)

    def get_acc(self):
        return self.acc



    def iou_batch(bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return (o)