"""Code for testing model"""

import torch
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
import torch.nn as nn
from utils import *
import sys

from yolo_v3 import YOLO_V3
from model_loss import MultiBox_Losss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nms_cpu(boxes, scores, overlap_threshold=0.3, min_mode=False):
    #boxes = boxes.cpu().numpy()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1 )
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep


def non_max_supression(image_pred,num_classes=1, overlap_thres = 0.4):

    class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
    # Iterate through all predicted classes
    unique_labels = detections[:, -1].cpu().unique()

    for c in unique_labels:
        # Get the detections with the particular class
        detections_class = detections[detections[:, -1] == c]
        # Sort the detections by maximum objectness confidence
        _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
        detections_class = detections_class[conf_sort_index]
        # Perform non-maximum suppression
        max_detections = []
        while detections_class.size(0):
            # Get detection with highest confidence and save as max detection
            max_detections.append(detections_class[0].unsqueeze(0))
            # Stop if we're at the last detection
            if len(detections_class) == 1:
                break
            # Get the IOUs for all boxes with lower confidence
            ious = IoU(max_detections[-1], detections_class[1:])
            # Remove detections with IoU >= NMS threshold
            detections_class = detections_class[1:][ious < overlap_threshold]

        max_detections = torch.cat(max_detections).data
    return max_detections

def non_max_suppression_fast(boxes, overlapThresh = 0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype('int')

def detect(image, obj_thres = 0.5):

    model = YOLO_V3(in_channels = 3, out_classes = 1)
    model.load_state_dict(torch.load('./best_chkpt.pth',map_location={'cuda:0': 'cpu'}))
    model = model.eval()

    transform = transforms.Compose([transforms.Resize((416,416)),transforms.ToTensor(),  \
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    #image = sys.argv[1]
    image = Image.open(img).convert('RGB')
    image = transform(image) #c,h,w
    image = Variable(image.float())

    predictions = model(image)

    priors11, priors8, priors6 = model.prior_anchors()
    anchors = torch.cat((priors11, priors8, priors6),0)
    anchors = Variable(anchors)

    strides = Variable(torch.from_numpy(np.array([32]*(3*13*13) + [16]*(3*26*26) + [8]*(3*52*52)))).float()

    pred = predictions.clone()
    pred[:,0] = (pred[:,0] + anchors[:,0])*strides  #cx
    pred[:,1] = (pred[:,1] + anchors[:,1])*strides  #cy
    pred[:,2] = torch.exp(pred[:,2]) * anchors[:,2] * strides  #w
    pred[:,3] = torch.exp(pred[:,3]) * anchors[:,3] * strides  #h

    pred = pred[(pred[:,4]>obj_thres).nonzero().squeeze(1).data] #selecting object classes

    boxes = cxcy_to_corners(pred.data[:,:4]) #centres to corners

    #valid anchor box indexes
    valid_index = torch.nonzero((boxes[:,0]>=0) &   \
        (boxes[:,1]>=0) &   \
        (boxes[:,2]<=416) &   \
        (boxes[:,3]<=416)).squeeze(1) 

    boxes = boxes[valid_index]

    #keep = nms_cpu(boxes.data.numpy(),scores.data.numpy())
    out = non_max_suppression_fast(boxes.data.numpy())
    print (out)

if __name__ == '__main__':
    detect(sys.argv[1])
