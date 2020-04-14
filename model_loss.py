"""Code for calculating model loss"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *

class MultiBox_Loss(nn.Module):

    def __init__(self,yolo_model,pos_iou_threshold,neg_iou_threshold,n_sample,pos_ratio):
        
        super(MultiBox_Loss,self).__init__()

        self.centre_priors = yolo_model.prior_anchors() #M,cx,cy,w,h
        self.corner_prioirs = cxcy_to_corners(self.centre_priors)
        self.strides = yolo_model.strides
        print (self.strides)
        self.n_out = self.centre_priors.shape[1]
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

        #Initiating loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def _process_targets_txty(self, targets, valid_index):

        anchors = Variable(self.centre_priors)[valid_index] #cx,cy,w,h
        strides = self.strides[valid_index]
        #print (strides.shape)

        """
        tx = (targets[:,0] - anchors[:,0])/anchors[:,2]
        ty = (targets[:,1] - anchors[:,1])/anchors[:,3]
        th = torch.log(targets[:,3]/anchors[:,3])
        tw = torch.log(targets[:,2]/anchors[:,2])

        targets = torch.cat((tx.unsqueeze(1),ty.unsqueeze(1), tw.unsqueeze(1),th.unsqueeze(1)),1)
        """

        tx = (targets[:,0]-anchors[:,0])/strides
        ty = (targets[:,1] - anchors[:,1])/strides
        th = torch.log(targets[:,3]/anchors[:,3])
        tw = torch.log(targets[:,2]/anchors[:,2])

        targets = torch.cat((tx.unsqueeze(1),ty.unsqueeze(1), tw.unsqueeze(1),th.unsqueeze(1)),1)
        
        return targets

    def _generate_targets(self, gt_boxes, gt_labels):

        corner_prioirs = self.corner_prioirs

        #valid anchor box indexes
        valid_index = torch.nonzero((corner_prioirs[:,0]>=0) &   \
            (corner_prioirs[:,1]>=0) &   \
            (corner_prioirs[:,2]<=416) &   \
            (corner_prioirs[:,3]<=416)).squeeze(1)  #N

        corner_gts = cxcy_to_corners(gt_boxes) #M,4

        valid_corner_priors = corner_prioirs[valid_index,:]

        IoUs = IoU(valid_corner_priors,corner_gts) #N,M

        #Idenfiying anchors with max overlap with groundtruth boxes
        gt_max_IoUs, gt_max_index = IoUs.max(dim = 0) #1,M
        gt_max_index = gt_max_index.squeeze(0).data

        #Identifying ground truth boxes for each anchor
        anchor_max_IoUs, anchor_max_index = IoUs.max(dim = 1) #N
        anchor_max_IoUs = anchor_max_IoUs.data

        #initiating tensor for assiging labels
        labels = torch.ones(valid_index.shape[0],)*-1

        #assigning positive label to anchors whose IoU with GT box is greater than threshold
        labels[anchor_max_IoUs >= self.pos_iou_threshold] = 1

        #assign negative labels to anchors whose IoU with GT box is greater than threshold
        labels[anchor_max_IoUs < self.neg_iou_threshold] = 0

        #assign positive labels to anchor boxes which have max overlap with GT boxes
        labels[gt_max_index] = 1

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
        anchor_gt_box = gt_boxes[anchor_max_index]  #10647,4 # cx,cy,w,h
        #anchor_gt_box = corner_gts[anchor_max_index]
        anchor_gt_labels = gt_labels[anchor_max_index] #10647

        targets = Variable(torch.zeros(len(valid_index),self.n_out))

        targets[:,:4] = self._process_targets_txty(anchor_gt_box, valid_index)
        targets[:,4] = labels

        gt_labels = gt_labels.data + 5

        for i, val in enumerate(anchor_max_index.data.numpy().tolist()) :
            #print (i,val, gt_labels[val].data[0])
            targets[i,gt_labels[val]] = 1

        return targets, valid_index

    def forward(self, predictions, gt_boxes,gt_labels) :

        targets, valid_index = self._generate_targets(gt_boxes, gt_labels) # N,cx,cy,w,h

        predictions = predictions.squeeze(0)[valid_index]

        obj_mask = (targets[:,4]==1).nonzero().squeeze(1).data
        non_obj_mask = (targets[:,4]==0).nonzero().squeeze(1).data

        pos_pred = predictions[obj_mask,:]
        neg_pred = predictions[non_obj_mask,:]
        
        pos_target = targets[obj_mask]
        neg_target = targets[non_obj_mask]

        loss_x = self.mse_loss(pos_pred[:,0],pos_target[:,0])
        loss_y = self.mse_loss(pos_pred[:,1],pos_target[:,1]) 
        loss_h = self.mse_loss(pos_pred[:,2],pos_target[:,2]) 
        loss_w = self.mse_loss(pos_pred[:,3],pos_target[:,3])

        try :

            loss_conf_obj = self.bce_loss(pos_pred[:,4],pos_target[:,4])
            loss_conf_nonobj = self.bce_loss(neg_pred[:,4],neg_target[:,4])
            loss_class = self.bce_loss(pos_pred[:,5:], pos_target[:,5:])

        except Exception as e:
            print (e)

        total_loss = (loss_x + loss_y + loss_w + loss_h) + (loss_conf_obj + loss_conf_nonobj + loss_class)

        return predictions,total_loss
