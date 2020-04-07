"""Code for calculating model loss"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *

class MultiBox_Loss(nn.Module):

    def __init__(self,yolo_model,pos_iou_threshold,neg_iou_threshold,n_sample,pos_ratio):
        
        super(MultiBox_Loss,self).__init__()

        self.priors = yolo_model.prior_anchors()
        self.n_out = self.priors.shape[1]
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

        #Initiating loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _generate_targets(self, gt_boxes, gt_labels):

        corner_prioirs = cxcy_to_corners(self.priors[:,:4]) #N,4
        corner_gts = cxcy_to_corners(gt_boxes) #M,4

        #print (corner_prioirs.shape, corner_gts.shape)

        IoUs = IoU(corner_prioirs,corner_gts) #N,M

        #Idenfiying anchors with max overlap with groundtruth boxes
        gt_max_IoUs, gt_max_index = IoUs.max(dim = 0) #1,M
        gt_max_index = gt_max_index.squeeze(0).data

        #Identifying ground truth boxes for each anchor
        anchor_max_IoUs, anchor_max_index = IoUs.max(dim = 1) #N
        anchor_max_IoUs = anchor_max_IoUs.data

        #print (anchor_max_index.shape)

        #initiating tensor for assiging labels
        labels = torch.ones(10647,)*-1

        #assign positive labels to anchor boxes which have max overlap with GT boxes
        labels[gt_max_index] = 1

        #assigning positive label to anchors whose IoU with GT box is greater than threshold
        labels[anchor_max_IoUs >= self.pos_iou_threshold] = 1

        #assign negative labels to anchors whose IoU with GT box is greater than threshold
        labels[anchor_max_IoUs < self.neg_iou_threshold] = 0
        #no of pos required
        n_pos = self.pos_ratio * self.n_sample
        #no of actual positives
        n_pos_actual = (labels==1).sum()

        if n_pos_actual > n_pos :
            #select random indexes of pos to ignore
            random_pos_index = (labels==1).nonzero()[torch.randperm(int(n_pos_actual-n_pos))].squeeze(1)
            labels[random_pos_index] = -1

        n_neg_actual = (labels==0).sum()

        n_neg_req = self.n_sample - (labels==1).sum()

        if n_neg_actual > n_neg_req :
            #select random indexes of negs to ignore
            random_neg_index = (labels==0).nonzero()[torch.randperm(n_neg_actual-n_neg_req)].squeeze(1)
            labels[random_neg_index] = -1

        #Assign locations to each anchor box
        anchor_gt_box = gt_boxes[anchor_max_index]  #10647,4
        #anchor_gt_labels = gt_labels[anchor_max_index] #10647

        targets = Variable(torch.zeros(10647,self.n_out))

        targets[:,:4] = anchor_gt_box
        targets[:,4] = labels

        gt_labels = gt_labels.data +5

        #targets[:,gt_labels] = 1

        for i, val in enumerate(anchor_max_index.data.numpy().tolist()) :
            #print (i,val, gt_labels[val].data[0])
            targets[i,gt_labels[val]] = 1

        return targets

    def forward(self, predictions, gt_boxes,gt_labels) :

        targets = self._generate_targets(gt_boxes, gt_labels)

        obj_mask = (targets[:,4]==1).nonzero().squeeze(1).data
        non_obj_mask = (targets[:,4]==0).nonzero().squeeze(1).data
        
        predictions = predictions.squeeze(0)
        #predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
        
        pos_pred = predictions[obj_mask,:]
        neg_pred = predictions[non_obj_mask,:]
        
        pos_target = targets[obj_mask]
        neg_target = targets[non_obj_mask]

        #print (pos_pred.shape, pos_target.shape)
        
        loss_x = self.mse_loss(pos_pred[:,0],pos_target[:,0])
        loss_y = self.mse_loss(pos_pred[:,1],pos_target[:,1])
        loss_w = self.mse_loss(pos_pred[:,2],pos_target[:,2])
        loss_h = self.mse_loss(pos_pred[:,3],pos_target[:,3])

        loss_conf_obj = self.bce_loss(pos_pred[:,4],pos_target[:,4])
        loss_conf_nonobj = self.bce_loss(neg_pred[:,4],neg_target[:,4])
        loss_class = self.bce_loss(pos_pred[:,5:], pos_target[:,5:])

        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_nonobj + loss_class

        return predictions,total_loss