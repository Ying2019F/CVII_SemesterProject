import os
import cv2
from sys import platform
import argparse
import subprocess 
import sys
import random
import math
import re
import time
import numpy as np
from models import *
from utils.datasets import *
from utils.utils import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import find_contours
from matplotlib import patches, lines 
from matplotlib.patches import Polygon
import IPython.display
from scipy import ndimage, misc
from PIL import Image 

# Importing code files necessary to run detections with maskrcnn 
from mrcnn import utils
import mrcnn.model as modellib
import cluster 

# Function to attain YOLOv3 detections

def detect(save_img=False):
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.yolo_weights, opt.half, opt.view_img, opt.save_txt
    img_size = (1024, 1024) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=(255,0,0))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

# Function to fuse detections 
def fusion(): 
    # Command Line Args 
    MASKRCNN_WEIGHTS_PATH = opt.maskrcnn_weights
    DEVICE = opt.device  # /cpu:0 or /gpu:0  
    IMAGE = opt.source 
    YOLO_WEIGHTS_PATH = opt.yolo_weights
    CONF_THRESHOLD = opt.conf_thres

    # Class Names 
    class_names = ['BG', 'cookie tin', 'book', 'plush duck', 'toy', 'remote', 'tennis ball', 'rubber duck', 'heart box', 'ping pong paddle', 'amazon', 'cat toy']


    # Running MaskRCNN 
    config = cluster.CustomConfig()
    ROOT_DIR = os.path.abspath("../../") # This could probably be a command line arg but thats a battle for later 
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                config=config)
    model.load_weights(MASKRCNN_WEIGHTS_PATH, by_name=True)

    # Getting MaskRCNN Detections
    image = matplotlib.image.imread(IMAGE)
    results = model.detect([image], verbose=1)
    r = results[0]

    # Converting YOLO detections to match structure and style of MaskRCNN detections
    try:
        f = open('output/' + str(IMAGE) + '.txt','r') 
    except FileNotFoundError:
        print("YOLO FOUND NO DETECTIONS, FUSION NOT POSSIBLE")
        f = open('filler' + '.txt','r') # If no detections found 

    content = f.read()
    allLines = content.split('\n')
    output = {} 
    output['xyxy']=[]
    output['cls']=[]
    output['score']=[]
    for singleLine in allLines:
        index = singleLine.split()
        list1 = index[0:4]
        output['xyxy'].append(list1)
        list2 = index[4:5]
        output['cls'].append(list2)
        list3 = index[5:6]
        output['score'].append(list3)

    # Convert yolo bounding box coordinates 
    list1 = output['xyxy']
    res = list(filter(None,list1))
    res = [[int((j)) for j in i] for i in res]
    res = np.array(res)
    A = res.shape[0]
    for i in range(A): 
        x1, y1, x2, y2 = res[i] #yolo has coordinates in a different order 
        res[i] = y1, x1, y2, x2

    r['rois'] = np.vstack((r['rois'],res))

    # Convert yolo class labels 
    classes = output['cls']
    cls_res = list(filter(None,classes))
    cls_res = [[str((j)) for j in i] for i in cls_res]
    d = {'0':9, '1':1, '2':10, '3':5, '4':3, '5':6, '6':2, '7':8, '8':7, '9':4, '10':11,}
    cls_res = [[d[j] for j in i] for i in cls_res] 
    cls_res = [[int((j)) for j in i] for i in cls_res]
    cls_res = np.array(cls_res)
    try:
        r['class_ids'] = np.vstack((r['class_ids'][:,None],cls_res))
    except ValueError:
        r['class_ids'] = np.vstack((r['class_ids'],cls_res))

    # Convert yolo scores 
    scores = output['score']
    scores_res = list(filter(None,scores))
    scores_res = [[float((j)) for j in i] for i in scores_res]
    scores_res = np.array(scores_res)
    try:
        r['scores'] = np.vstack((r['scores'][:,None],scores_res))
    except ValueError:
        r['scores'] = np.vstack((r['scores'],scores_res))

    roi_scores = r['scores']
    roi_class_names = r['class_ids']
    boxes = r['rois']
    keep = np.where(roi_class_names > 0)[0]
    keep = np.intersect1d(keep, np.where(roi_scores >= CONF_THRESHOLD)[0]) 

    #Performing non max suppression for all detections 
    pre_nms_boxes = boxes[keep]
    pre_nms_classes = roi_class_names[keep]
    pre_nms_scores = np.squeeze(roi_scores[keep])

    nms_keep = []
    for class_id in np.unique(pre_nms_classes):
        # Pick detections of this class
        ixs = np.where(pre_nms_classes == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                                pre_nms_scores[ixs],
                                                opt.iou_thres)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)

    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))

    ixs = np.arange(len(keep))
    image = Image.open(IMAGE)
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for i, idx in enumerate(ixs): 
        y1, x1, y2, x2 = boxes[keep][idx]
        p = patches.Rectangle((x1,y1),x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="solid", edgecolor='b',facecolor='none')
        ax.add_patch(p)
        # add labels 
        class_id = roi_class_names[keep][idx]
        score = roi_scores[keep][idx] 
        label = class_names[int(class_id)]
        caption = "{} {:.3f}".format(label,float(score)) if score else label
        ax.text(x1,y1 + 8, caption, color='w', size=8, backgroundcolor="none")
    
    fig.savefig('fusionDetection.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='best_mobile_500.pt', help='yolo weights path')
    parser.add_argument('--device', type=str, default='', help='device cpu or gpu (/gpu:0)')
    parser.add_argument('--maskrcnn_weights', type=str, default='mask_rcnn_object_0050.h5', help='maskrcnn weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path for yolo')
    parser.add_argument('--names', type=str, default='data/yolo.names', help='*.names path')
    opt = parser.parse_args()
    print(opt)
    print("=====RUNNING YOLOV3=====")
    detect()
    print("=====FUSING RESULTS=====")
    fusion()
    



