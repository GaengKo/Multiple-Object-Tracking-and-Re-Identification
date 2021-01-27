import numpy as np


"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]we2
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""
class load_label:
    fname = ""
    def load_txt(self, fname):
        result = []
        image_num = -1
        self.fname = fname
        f= open('C:/Users/YK/Desktop/dataset/label_02/'+fname,'r')
        for s in f:
            fline = s.split()
            if image_num < int(fline[0]):
                temp = []
                temp.append(fline[1:-1])
                result.append(temp)
                image_num = image_num+1
            else:
                result[int(fline[0])].append(fline[1:-1])
        print(len(result[0]))
        print(len(result))
        f.close()

        return result




