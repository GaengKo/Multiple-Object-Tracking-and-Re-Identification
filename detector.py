from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *


class Yolo():
    def __init__(self):
        self.weights = 'weights/best.pt'
        self.half = True
        self.cfg = 'cfg/yolov3-spp.cfg'
        self.imgsz = (320, 192) if ONNX_EXPORT else 512  # (320, 192) or (416, 256) or (608, 352) for (height, width)

        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else '')

        self.model = Darknet(self.cfg, self.imgsz)

        # load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        self.model.to(self.device).eval()

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.names = 'data/kitti_OD.names'
        self.names = list(glob.iglob('./**/' + self.names, recursive=True))[0]
        #print('***',end='')

        self.names = load_classes(self.names)
        #print(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
    def forword(self, frame):
        im0 = letterbox(frame, new_shape=self.imgsz)[0]
        im0 = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im0 = np.ascontiguousarray(im0)

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img.float()) if self.device.type != 'cpu' else None  # run once
        img = torch.from_numpy(im0).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()
        #print(img.shape)
        pred = self.model(img, augment=True)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.6,
                                   multi_label=False, agnostic=True)

        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                #print(det[:,:4])
        try:
            result = pred[0].cpu().detach().numpy()
        except Exception as E:
            print(pred)
            return []
        #print(result)
        return result