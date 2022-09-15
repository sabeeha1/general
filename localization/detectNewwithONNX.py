import argparse
import os
# import platformimport shutil
import time
from pathlib import Path
import onnxruntime as ort
import cv2
import torch

from numpy import random


from utils.datasets import LoadImages
from utils.general import (non_max_suppression, xyxy2xywh,save_one_box,increment_path)
from utils.plots import plot_one_box

from utils.datasets import *
from sort_master.sort import *


# def new_coord(x,x_correction, y_correction):
def new_coord(img_size, x,img_original_size):
    ##Correct the coordinates for the test images
    y_correction = img_original_size[0]/img_size[0]
    x_correction = img_original_size[1]/img_size[0]

    ##I first normalize the coordinates then adjust them as per new coordinates
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] * x_correction )- 5  # x-start (top left)
    y[:, 2] = (x[:, 2] * x_correction ) + 5 # x-end (bottom right)
    
    y[:, 1] = ( x[:, 1] * y_correction ) - 5 # y-start (top left)
    y[:, 3] = ( x[:, 3] * y_correction )+ 5# y-end (bottom right)    
    return y

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names,save_crop = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names, opt.save_crop
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = None
    if torch.cuda.is_available():
        half = device.type != 'cpu'  # half precision only supported on CUDA



    ort_session = ort.InferenceSession("/home/neural/Documents/Sabeeha/codes/localization/weights/groc60.pt")
    print("Loaded onnx model.\n")

    save_dir = increment_path(Path(out) / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Second-stage classifier
    save_txt = True
    save_img = True

    # Set Dataloader
    vid_path, vid_writer = True, True
    save_img = True
    dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    
        
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        if torch.cuda.is_available():
            img = img.half() if half else img.float()  # uint8 to fp16/32
        else:
            img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # img_numpy = img.numpy()
        
        img_4_onnx = cv2.resize(im0s,(imgsz, imgsz))
        img_4_onnx = img_4_onnx[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_4_onnx = np.ascontiguousarray(img_4_onnx)
        img_4_onnx = torch.from_numpy(img_4_onnx).to(device)
        img_4_onnx = img_4_onnx.float()
        img_4_onnx /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_4_onnx.ndimension() == 3:
            img_4_onnx = img_4_onnx.unsqueeze(0)
#         ## ONNX model uses numpy as input, so convert to numpy before sending as input
        img_numpy = img_4_onnx.numpy()
        # Inference
        t1 = time.time()
        pred = torch.as_tensor(ort_session.run(None,({ort_session.get_inputs()[0].name:img_numpy}))[0])     
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time.time()
        
        i = 0

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # print(img.shape[2:])
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            track_bbs_ids = mot_tracker.update(det)

            track_bbs_ids = torch.from_numpy(track_bbs_ids)
            if track_bbs_ids is not None and len(track_bbs_ids):
                # Rescale boxes from img_size to im0 size
                # track_bbs_ids[:, :4] = scale_coords(img.shape[2:], track_bbs_ids[:, :4], im0.shape).round()
                track_bbs_ids[:, :4] = new_coord(img.shape[2:], track_bbs_ids[:, :4], im0.shape).round()

                for *xyxy, track_id in track_bbs_ids:
                    track_num = int(track_id.item())
                
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (track_num, *xywh))  # label format
                        
                    if save_img or view_img:  # Add bbox to image
                        # label = '%s' (names[track_num])
                        label = str(track_num)
                        plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)
                        if save_crop:
                            pass
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / str(track_num) / f'{Path(p).stem}.jpg', BGR=True)

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
# =============================================================================
#             if view_img:
#                 cv2.imshow(p, im0)
#                 if cv2.waitKey(1) == ord('q'):  # q to quit
#                     raise StopIteration
# =============================================================================

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    # if save_txt or save_img:
    #     print('Results saved to %s' % Path(out))
    #     if platform == 'darwin' and not opt.update:  # MacOS
    #         os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/groc.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--save-crop', default=True, action='store_true', help='save cropped prediction boxes')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        mot_tracker = Sort()
        detect()
