
import argparse
import os
import re
import shutil
import time
import glob
import sys
from pathlib import Path
import pkg_resources as pkg
import cv2
import torch
import math
import logging
import torch.nn as nn
import numpy as np
import torchvision
import onnxruntime
import warnings
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.15, iou_thres=0.40, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger

    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging('yolov5')  # define globally (used in train.py, val.py, detect.py, etc.)

def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='.\weights.onnx', device=None, dnn=False, data=None):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        onnx = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = str(Path(str(w).strip().replace("'", '')))
        if data:  # data.yaml path (optional)
            names = data

        LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
        cuda = torch.cuda.is_available()
        #check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference

        b, ch, h, w = im.shape  # batch, channel, height, width
        im = im.cpu().numpy()  # torch to numpy

        y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]

        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    @staticmethod
    def model_type(p='path/to/model.onnx'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = ['.onnx']
        #check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        onnx = (s in p for s in suffixes)
        return onnx

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (dw, dh)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni # number of files
        self.video_flag = [False] * ni
        self.mode = 'image'
        self.auto = auto
        self.cap = None
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def __len__(self):
        return self.nf  # number of files

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    save = False
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)
    return crop


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def reduced_frames(extracted_frames_dir, frame_count, video, file_number):
        VideoFrame = cv2.VideoCapture(video)  # Creating a VideoCapture instance to read video
        frameCount = 0  # I do not have to read the full video, so I am setting a count for video frames
        length = int(VideoFrame.get(cv2.CAP_PROP_FRAME_COUNT))
        if length==0:
            print(f"No frames in {video} file or file name is incorrect")
            return False
        frame_gap = round(length / frame_count,2)
        readingcheck = 1
        frame_id = 0

        while (readingcheck):  # Use this loop if you want to read until the end of the recorded video
            readingcheck, frame = VideoFrame.read()
            gap_required = int(round(frame_gap*frame_id,0))

            if frameCount == gap_required:

                file_number_str =  str(file_number).zfill(3) + '.jpg'
                file_path = os.path.join(extracted_frames_dir, file_number_str)
                if (readingcheck):

                    cv2.imwrite(file_path, frame)
                    file_number += 1
                    frame_id +=1

            frameCount = frameCount + 1

        return file_number

def image_resize(image,MAX,crop):
    ratiox, ratioy = float(image.shape[0]) / MAX, float(image.shape[1]) / MAX
    if ratiox > ratioy:

        finalx = image.shape[0] / ratiox
        finaly = image.shape[1] / ratiox
    else:
        finalx = image.shape[0] / ratioy
        finaly = image.shape[1] / ratioy

    resized = cv2.resize(image, (int(finaly),int(finalx)), interpolation = cv2.INTER_AREA)
    if crop:
        resized = cv2.resize(image, (int(finaly), int(finalx/2)), interpolation=cv2.INTER_AREA)
    return resized

#Generate 10 files from one image
def Data_augmentation(img,num,data_aug_dir,rotate,crop,MAX):
        #Decreasing brightness
        for i in [0,70,40]:
            bright = np.ones(img.shape, dtype='uint8') * i
            dim_file = cv2.subtract(img, bright)
            dim_file = image_resize(dim_file, MAX,crop)
            file_name = str(num).zfill(4) + '.jpg'
            cv2.imwrite(os.path.join(data_aug_dir,file_name), dim_file)
            num += 1

        #Increasing brightness
        for i in [5,10]:
             for j in [10]:
                 bright = (np.ones(img.shape, dtype='uint8') * i) + j
                 bright_file = cv2.add(img,bright)
                 file_name = str(num).zfill(4)+'.jpg'
                 bright_file = image_resize(bright_file, MAX,crop)
                 cv2.imwrite(os.path.join(data_aug_dir,file_name) ,bright_file)
                 num +=1

        #No crop if said
        if  crop:
            height,width,_=img.shape

            #crop from right
            cropped_file = img[:, 0:int(width / 2)]
            cropped_file = image_resize(cropped_file, MAX,crop)
            file_name =  str(num).zfill(4) + '.jpg'
            cv2.imwrite(os.path.join(data_aug_dir, file_name), cropped_file)
            num += 1
            # crop from left
            cropped_file = img[:, int(width / 2):]
            cropped_file = image_resize(cropped_file, MAX,crop)
            file_name =  str(num).zfill(4) + '.jpg'
            cv2.imwrite(os.path.join(data_aug_dir, file_name), cropped_file)
            num += 1

        #No rotate if said
        if rotate:
            M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), 180, 1.0)
            rotated180 = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
            rotated180 = image_resize(rotated180, MAX,crop)
            file_name = str(num).zfill(4)+'.jpg'
            cv2.imwrite(os.path.join(data_aug_dir,file_name), rotated180)
            #print("Data_augmentation", file_name)
            num += 1

        #creating blurriness
        for i in [(8,8),(11,11)]:
             img_blurr = cv2.blur(img, i)
             img_blurr = image_resize(img_blurr, MAX,crop)
             file_name = str(num).zfill(4)+'.jpg'
             cv2.imwrite(os.path.join(data_aug_dir,file_name) ,img_blurr)
             num +=1
        return num

@torch.no_grad()
def run(input_dir,  # save results to project/name
        output = 'reference' ,  # save results to project/name
        frame_count = 50, #Reduced frames extracted from source video
        rotate=False,
        crop=False,
        final_size = 200,
        weights= 'weights.onnx'
        ):
        'Creating Directories for storing frames, trimmed images and the dataset '
        vid_extensions = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv']
        listVideos = os.listdir(input_dir)

        listVideos = [f for f in listVideos
                      if any(f.lower().endswith(ext) for ext in vid_extensions)]

        if  not os.path.exists(output):
            os.makedirs(output)
        #intiating max_det=1000
        imgsz = [640,640]
        max_det = 1000
        classes = None
        agnostic_nms = False
        exist_ok = False
        conf_thres = 0.15  # confidence threshold
        dnn = False
        iou_thres = 0.40  # NMS IOU threshold
        file_num = 0

        output = os.path.join(output,os.path.basename(input_dir))
        data_aug_dir = os.path.join(output, 'Augmented_Frames')
        extracted_frames_dir = os.path.join(output, 'Selected_Frames')

        if os.path.exists(extracted_frames_dir):
            shutil.rmtree(extracted_frames_dir)
        os.makedirs(extracted_frames_dir)
        if  os.path.exists(data_aug_dir):
            shutil.rmtree(data_aug_dir)
        os.makedirs(data_aug_dir)

        for video in listVideos:
            print("Extracting frames from ",video,"!")
            file_num = reduced_frames(extracted_frames_dir, frame_count,os.path.join(input_dir,video),file_num)


        print("<============making final Dataset============>")
        file_number = 0
        source = extracted_frames_dir
        device = 'cpu'
        model = DetectMultiBackend(weights, device= device, dnn=dnn, data= ['item'])

        stride, names, onnx = model.stride, model.names, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        half = False
        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=onnx)
        bs = 1  # batch_size

        # Run inference
        Dataset_image_num = 0
        dt, seen,j = [0.0, 0.0, 0.0], 0,0

        #Here im is the processed image while im0s is the original image
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy()   # for save_crop

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Change Done,extracting the biggest box
                    big = 0
                    count= 0
                    area=0
                    flag= 0
                    centre_x,centre_y = im0.shape[1]/2, im0.shape[0]/2
                    for *xyxy, conf, cls in det:
                        left,top,right,bottom = xyxy
                        area = (right-left)*(bottom-top)
                        #Keep the box with maximum area
                        if ((left<= centre_x <= right) and (top<= centre_y <= bottom)) and (int(area) > big):
                            flag=1
                            big = area
                            l, t, r, b = int(left.item()), int(top.item()), int(right.item()), int(bottom.item())
                            det[count][-1] = 0
                            final = [det[count]]
                        count +=1

                    if flag==1:
                        det_crop = im0[t-100:t,l:r]
                        imgtop_crop = im0[0:100,l:r]

                        det_crop_color_sum = np.sum(det_crop, axis=1) // 100
                        det_crop_color_sum = np.sum(det_crop_color_sum, axis=0)
                        det_crop_color_sum = det_crop_color_sum //(r-l)
                        det_crop_color_sum = np.sum(det_crop_color_sum//3).item()

                        imgtop_crop_color_sum = np.sum(imgtop_crop, axis=1) // 100
                        imgtop_crop_color_sum = np.sum(imgtop_crop_color_sum, axis=0)
                        imgtop_crop_color_sum = imgtop_crop_color_sum //(r-l)
                        imgtop_crop_color_sum = np.sum(imgtop_crop_color_sum//3).item()

                        #If image top color is much different from base, Don't save the detection. As it might have object above also
                        if (abs(imgtop_crop_color_sum-det_crop_color_sum)>15):
                            continue

                        det = det.tolist()
                        det.clear()
                        det = final.copy()
                        final.clear()
                        file_number += 1
                        for *xyxy, conf, cls in reversed(det):
                            file_n = os.path.join(str(file_number)+'.jpg')
                            #it first convert the detection coordinates into original imge size and then crop the box from the image
                            crop_img = save_one_box(xyxy, imc, file_n, BGR=True)
                            Dataset_image_num = Data_augmentation(crop_img, Dataset_image_num,data_aug_dir,rotate,crop,final_size)  # change in contrast

                    j+=1
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

def parse_opt():
    parser = argparse.ArgumentParser(description='Dataset development from TurnTable videos.')
    parser.add_argument('input_dir', help='Path to input turn table videos directory')
    parser.add_argument('--output',type=str, default='reference', help='Path to output dataset directory, Default: inference')
    parser.add_argument('--frame-count', type=int, default=50, help='No of extracted intermediate frames from video for data augmentation, Default: 50')
    parser.add_argument('--final-size', type=int, default=200, help='Final size of images in the reference dataset, Default: 200')
    parser.add_argument('--rotate', action='store_true', help='rotate image 180-degrees while performing data augmentation, Defalut: False')
    parser.add_argument('--crop', action='store_true',help='crop image while performing data augmentation,  Default: False')
    parser.add_argument('--weights', default='weights.onnx',help='Path to neural network weights file, Default: weights.onnx')

    opt = parser.parse_args()
    #print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()


    main(opt)

