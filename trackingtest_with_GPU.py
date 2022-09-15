import argparse
from filterpy.kalman import KalmanFilter
import numpy as np
import os
import torch
import time
import shutil
import torchvision
import glob
import cv2
import onnxruntime as ort
from numpy import random
from pathlib import Path
import re
import json

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2startingxywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if x[:, 0] > 0:
        y[:, 0] = x[:, 0]  # x corner
    else:
        y[:, 0] = 1
    if x[:, 1] > 0:
        y[:, 1] = x[:, 1]  # y corner
    else:
        y[:, 1] = 1
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def save_one_box(xyxy, im, file='image.jpg', gain=1.0, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)

    return crop


def save_one_box_withoutIncrement(videoName, frame_count, trackID, xyxy, im, path, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        extra = videoName.split('.')[0]
        image_path = os.path.join(path, extra + 'frame' + str(frame_count) + 'trackID' + str(trackID) + '.jpg')
        cv2.imwrite(image_path, crop)

    return crop


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

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
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, auto_size=32):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        # images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        print(videos)
        nv = len(videos)
        print(videos, "******************************", len(videos))

        self.img_size = img_size
        self.auto_size = auto_size
        self.files = videos
        self.nf = nv  # number of files
        self.video_flag = [True] * nv
        self.mode = 'images'
        if any(videos):
            self.frame = 0
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, vid_formats)
        # print(self.frame)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # print(self.frame)
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
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


def new_coord(img_size, x, img_original_size, bbox_margin):
    ##Correct the coordinates for the test images
    y_correction = img_original_size[0] / img_size[0]
    x_correction = img_original_size[1] / img_size[0]

    ##I first normalize the coordinates then adjust them as per new coordinates
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] * x_correction) - bbox_margin  # x-start (top left)
    y[:, 2] = (x[:, 2] * x_correction) + bbox_margin  # x-end (bottom right)
    y[:, 1] = (x[:, 1] * y_correction) - bbox_margin  # y-start (top left)
    y[:, 3] = (x[:, 3] * y_correction) + bbox_margin  # y-end (bottom right)
    return y


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def detect():
    out, source, weights, bbox_margin = opt.output, opt.source, opt.weights, opt.box_margin
    imgsz, save_txt, save_video = opt.img_size, opt.save_bbox_txt, opt.save_localized_video
    # imgsz =640

    save_crop = True
    print(save_txt, save_video, save_crop)
    color = [random.randint(0, 255) for _ in range(3)]
    # Find or create output directory
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = None

    listVideos = os.listdir(source)

    listVideos = [f for f in listVideos
                  if any(f.lower().endswith(ext) for ext in vid_formats)]

    for video in listVideos:
        video_name = os.path.splitext(video)
        video_dataset = os.path.join(out, video_name[0])

        if os.path.exists(video_dataset):
            shutil.rmtree(video_dataset)
        os.mkdir(video_dataset)

    # Loading onnx weights
    ExecutionProviders = [["CUDAExecutionProvider"], ["CPUExecutionProvider"]]
    if ort.get_device() == 'GPU':

        ort_session = ort.InferenceSession(weights, providers=ExecutionProviders[0])
        device = 'cuda'
    else:
        ort_session = ort.InferenceSession(weights, providers=ExecutionProviders[1])
        device = 'cpu'

    # ort_session = ort.InferenceSession(weights)
    print("Loaded onnx model in: \n", ort.get_device())

    # Set Dataloader
    vid_path, vid_writer = True, True
    dataset = LoadImages(source, img_size=imgsz, auto_size=64)
    save_img = True

    dataset = LoadImages(source, img_size=imgsz, auto_size=64)
    for index, videoName in enumerate(listVideos):

        KalmanBoxTracker.count = 0
        video_name = os.path.join(source, videoName)
        print(video_name)

        dataset = LoadImages(video_name, img_size=imgsz, auto_size=64)
        # print(dataset.__len__)
        # break

        # Run inference
        t0 = time.time()

        frame_count = 0
        # trackIDcheck = 1
        for path, img, im0s, vid_cap in dataset:
            vid_name = (os.path.basename(path))
            vid_name_base = os.path.splitext(vid_name)[0]
            save_dir = os.path.join(out, vid_name_base)

            # img = torch.from_numpy(img).to(device)
            # if torch.cuda.is_available():
            #     img = img.half() if half else img.float()  # uint8 to fp16/32
            # else:
            #     img = img.float()
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)

            img_4_onnx = cv2.resize(im0s, (imgsz, imgsz))
            img_4_onnx = img_4_onnx[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img_4_onnx = np.ascontiguousarray(img_4_onnx)
            img_4_onnx = torch.from_numpy(img_4_onnx).to('cpu')
            img_4_onnx = img_4_onnx.float()
            img_4_onnx /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_4_onnx.ndimension() == 3:
                img_4_onnx = img_4_onnx.unsqueeze(0)
            #         ## ONNX model uses numpy as input, so convert to numpy before sending as input
            img_numpy = img_4_onnx.cpu().numpy()
            # img_numpy =torch.Tensor(img_numpy)
            # print(type(img_4_onnx),img_4_onnx)
            t1 = time.time()

            pred = torch.as_tensor(ort_session.run(None, ({ort_session.get_inputs()[0].name: img_numpy}))[0])

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=False,
                                       classes=None, agnostic=False)
            t2 = time.time()

            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s = s + vid_name + " Frame" + str(frame_count)
                s += ' %gx%g ' % img_4_onnx.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                original_image = im0.copy()
                imc = im0.copy() if save_crop else im0  # for save_crop
                track_bbs_ids = mot_tracker.update(det)
                track_bbs_ids = torch.from_numpy(track_bbs_ids)
                # print(track_bbs_ids)

                if track_bbs_ids is not None and len(track_bbs_ids):
                    # Rescale boxes from img_size to im0 size
                    track_bbs_ids[:, :4] = new_coord(img_numpy.shape[2:], track_bbs_ids[:, :4], im0.shape,
                                                     bbox_margin).round()
                    view_img = True
                    for *xyxy, track_id in track_bbs_ids:
                        track_num = int(track_id.item())

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh (centerx, centery, w, h)
                            # xywh = (xyxy2startingxywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # original xywh  (cornerx, cornery, w h)
                            # xywh4json = (xyxy2startingxywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # original xywh  (cornerx, cornery, w h)
                            txt_path = save_dir + 'frame' + str(frame_count)
                            # print(xywh, type(xywh))
                            with open(txt_path + 'Original.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (track_num, *xywh))  # label format
                                # f.write(('%3d '+'%.3f ' * 4 + '\n') % (track_num, *xywh))  # integer track id, then normalzied x,y,w,h with 3 decimal places
                        if view_img:  # Add bbox to image
                            label = str(track_num)

                            plot_one_box(xyxy, im0, label=label, color=color, line_thickness=3)
                            # print(save_dir)
                            if save_crop:
                                cropped_image_path = os.path.join(save_dir, 'dataset')
                                if not os.path.exists(cropped_image_path):
                                    os.mkdir(cropped_image_path)
                                cropped_image_path = os.path.join(cropped_image_path, str(track_num))
                                # dir = os.path.join(dir,f'{Path(p).stem}.jpg')
                                if not os.path.exists(cropped_image_path):
                                    os.mkdir(cropped_image_path)

                                # save_one_box(frame_count,track_id, xyxy, imc, pad=bbox_margin, file= dir, BGR=True)
                                save_one_box_withoutIncrement(videoName, frame_count, track_num, xyxy, imc,
                                                              cropped_image_path, BGR=True, save=True)

                print('%sDone. (%.3fs)' % (s, t2 - t1))

                if save_video:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_name_new = os.path.join(save_dir, vid_name_base + '.mp4')
                        vid_writer = cv2.VideoWriter(vid_name_new, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
                frame_count = frame_count + 1
        print('Done. (%.3fs)' % (time.time() - t0))


class Sort(object):
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.1):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # Maximum number of frames to keep alive a track without associated detections.
        self.min_hits = min_hits  # Minimum number of associated detections before track is initialised.
        self.iou_threshold = iou_threshold  # Minimum IOU for match.
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
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
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

        def __del__(self):
            print("Object gets destroyed");


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

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
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


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # [[0.         0.96700392 0.         ... 0.         0.         0.        ],[             ]upto all cross matchings]
    iou_matrix = iou_batch(detections, trackers)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset development from tracking videos")
    parser.add_argument('source', type=str, help='Path to input videos directory')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='reference',
                        help='Path to output dataset directory, Default: reference')  # output folder
    parser.add_argument('--weights', type=str, default='trackingvideoYOLOR.onnx',
                        help=' Neural network weights file, Default: trackingvideoYOLOR.onnx')
    parser.add_argument('--box-margin', type=int, default=0, help=' Bounding box margin in pixels, Default: 0')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels), Default: 640')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold, Default: 0.1')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS, Default: 0.1')
    parser.add_argument('--save-bbox-txt', type=bool, default=False,
                        help='to save localised boxes details, Default: True')
    parser.add_argument('--save-localized-video', type=bool, default=True,
                        help='save video with localised boxes, Default: True')

    opt = parser.parse_args()

    with torch.no_grad():
        mot_tracker = Sort()
        detect()