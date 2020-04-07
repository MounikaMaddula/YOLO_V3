"""Code for all general functions"""

import pandas as pd 
import numpy as np 
import torch
from torch.autograd import Variable

def cxcy_to_corners(input_tensor):
    """
    Converts cx,cy,h,w tensor to bbox tensor of shape xmin,ymin,xmax,ymax

    Arguments:
        input - tensor of shape N,cx,cy,h,w
        output - tensor of shape N,xmin,ymin,xmax,ymax
    """
    xmin = input_tensor[:,0] - input_tensor[:,2]/2
    xmax = input_tensor[:,0] + input_tensor[:,2]/2
    ymin = input_tensor[:,1] - input_tensor[:,3]/2
    ymax = input_tensor[:,1] + input_tensor[:,3]/2

    output = Variable(torch.zeros(input_tensor.shape[0],4))
    output[:,0] = xmin
    output[:,1] = ymin
    output[:,2] = xmax
    output[:,3] = ymax

    return output


def corners_to_cxcy(input_tensor):
    """
    Converts cx,cy,h,w tensor to bbox tensor of shape xmin,ymin,xmax,ymax

    Arguments:
        input - tensor of shape N,xmin,ymin,xmax,ymax
        output - tensor of shape N,cx,cy,h,w
    """
    cx = (input_tensor[:,0] + input_tensor[:,2])/2
    cy = (input_tensor[:,1] + input_tensor[:,3])/2
    h = input_tensor[:,3] - input_tensor[:,1]
    w = input_tensor[:,2] - input_tensor[:,0]

    output = torch.zeros(input_tensor.shape[0],4)
    output[:,0] = cx
    output[:,1] = cy
    output[:,2] = h
    output[:,3] = w

    return output

def intersection(box_1,box_2):
    """
    Computes the intersection area of 2 tensors

    Arguments:
        box_1 - tensor of shape N,xmin,ymin,xmax,ymax
        box_2 - tensor of shape M,xmin,ymin,xmax,ymax

        Output - tensor of shape N,M
    """
    N = box_1.shape[0]
    M = box_2.shape[0]

    xy_min = torch.max(box_1[:,:2].unsqueeze(1).expand(N,M,2), box_2[:,:2].unsqueeze(0).expand(N,M,2))
    xy_max = torch.min(box_1[:,2:].unsqueeze(1).expand(N,M,2), box_2[:,2:].unsqueeze(0).expand(N,M,2))

    width = torch.clamp(xy_max[:,:,0]-xy_min[:,:,0],min = 0)
    height = torch.clamp(xy_max[:,:,1]-xy_min[:,:,1],min = 0)

    return width*height


def IoU(box_1,box_2):
    """
    Computes the intersection area of 2 tensors

    Arguments:
        box_1 - tensor of shape N,xmin,ymin,xmax,ymax
        box_2 - tensor of shape M,xmin,ymin,xmax,ymax

        Output - Intersection over union area of both boxes
                 Shape - N,M
    """   

    N = box_1.shape[0]
    M = box_2.shape[0]

    box1_wh =  box_1[:,2:] - box_1[:,:2]
    box1_area = (box1_wh[:,0]*box1_wh[:,1]).unsqueeze(1).expand(N,M)

    box2_wh =  box_2[:,2:] - box_2[:,:2]
    box2_area = (box2_wh[:,0]*box2_wh[:,1]).unsqueeze(0).expand(N,M)

    inter = intersection(box_1,box_2)

    #print (box1_area.data, box2_area.data, inter,(box1_area + box2_area).data, (box1_area + box2_area- inter).data )

    iou = inter/(box1_area + box2_area - inter)

    return iou

def NMS(detections,keep_thres = 0.3,overlap_threshold = 0.5):

    #print (detections.shape) #N,10647,cx,cy,w,h,80
    detections = detections.squeeze(0) #10647,85
    #print (detections.shape)

    detections[:,4:] = torch.sigmoid(detections[:,4:])

    scores = detections[:,4] #10647,

    _,indx = scores.sort(0)

    indx = list(indx.numpy())[:40]

    req_boxes = []

    for i, val in enumerate(indx) :
        #print (detections[val,4])
        if detections[val,4] > keep_thres :
            #print (detections[val,:4].shape)
            box1 = cxcy_to_corners(detections[val,:4].unsqueeze(0))
            req_boxes.append(val)
            for next_ind in indx[i+1:]:
                box2 = cxcy_to_corners(detections[next_ind,:4].unsqueeze(0))
                #print (IoU(box1,box2), intersection(box1,box2))
                if IoU(box1,box2).data[0][0] > overlap_threshold :
                    detections[next_ind,4] = 0

    final = torch.cat([detections[x].unsqueeze(0) for x in req_boxes],0)

    final_coords = final[:,:4]
    _,final_classes = final[:,5:].max(1)

    return final_coords, final_classes
