"""Code for calculating model loss"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiBox_Loss(nn.Module):

    def __init__(self,yolo_model,pos_iou_threshold,neg_iou_threshold,n_sample,pos_ratio):
        
        super(MultiBox_Loss,self).__init__()

        self.centre_priors11,self.centre_priors8, self.centre_priors6 = yolo_model.prior_anchors() #M,cx,cy,w,h #priors11,priors8,priors6
        #self.corner_prioirs = cxcy_to_corners(self.centre_priors)
        #self.strides = yolo_model.strides
        #print (self.strides)
        self.n_out = self.centre_priors11.shape[1]
        #self.n_out = 5 + yolo_model.out_classes
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

        #Initiating loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def _generate_targets(self,grid_size, centre_prioirs, gt_boxes, gt_labels):

        gt_boxes = gt_boxes*grid_size

        #targets = Variable(torch.zeros(grid_size*grid_size*3,6)).to(device)
        #pos_labels = torch.ones(centre_prioirs.shape[0]).to(device)*-1
        #neg_labels = torch.zeros(centre_prioirs.shape[0]).to(device)

        targets = Variable(torch.zeros(grid_size*grid_size*3,6))
        pos_labels = torch.zeros(centre_prioirs.shape[0])
        neg_labels = torch.ones(centre_prioirs.shape[0])

        for box_ind, box in enumerate(gt_boxes) :
            box = box.unsqueeze(0)
            gt_i, gt_j = box[:,:2].long().t()
            gt_w,gt_h = box[:,2:4].t()

            #gt_anchor_box = torch.zeros(1,4).to(device)
            gt_anchor_box = torch.zeros(1,4)
            gt_anchor_box[:,2:] =  box[:,2:4].data
            #print (gt_j,gt_i)
            anc_ind = int(39*gt_j) + (3*gt_i)
            anc_ind = anc_ind.data[0]
            #print (anc_ind)
            anchors_wh = centre_prioirs[anc_ind:anc_ind+3][:,2:4]
            #anchors = torch.zeros(3,4).to(device)
            anchors = torch.zeros(3,4)
            anchors[:,2:] = anchors_wh
            ious = IoU(gt_anchor_box,anchors)
            try :
                ignore = (ious>0.3).nonzero().squeeze(1).data
                neg_labels[anc_ind+ignore] = 0
            except :
                a = 1
            best_n = np.argmax(ious)

            targets[anc_ind+best_n,2] = torch.log(box[:,2]/centre_prioirs[anc_ind+best_n,2])
            targets[anc_ind+best_n,3] = torch.log(box[:,3]/centre_prioirs[anc_ind+best_n,3])
            targets[anc_ind+best_n,:2] = box[:,:2] - torch.floor(box[:,:2])
            targets[anc_ind+best_n,4] = 1
            targets[anc_ind+best_n,gt_labels[box_ind].data[0]+5] = 1
            neg_labels[anc_ind+best_n] = 0
            pos_labels[anc_ind+best_n] = 1

        return targets, pos_labels, neg_labels


    def forward(self, predictions, gt_boxes,gt_labels) :

        #gt_boxes = corners_to_cxcy(gt_boxes)

        targets11, pos_labels11, neg_labels11 = self._generate_targets(13, self.centre_priors11, gt_boxes, gt_labels) # N,cx,cy,w,h
        targets8, pos_labels8, neg_labels8 = self._generate_targets(26, self.centre_priors8, gt_boxes, gt_labels)
        targets6, pos_labels6, neg_labels6 = self._generate_targets(52, self.centre_priors6, gt_boxes, gt_labels)

        predictions = predictions.squeeze(0)
        
        targets = torch.cat((targets11,targets8,targets6),0)
        neg_labels = torch.cat((neg_labels11,neg_labels8,neg_labels6),0)
        pos_labels = torch.cat((pos_labels11,pos_labels8,pos_labels6),0)

        obj_mask = (targets[:,4]==1).nonzero().squeeze(1).data
        non_obj_mask = neg_labels - pos_labels
        non_obj_mask = (non_obj_mask==1).nonzero().squeeze(1)

        pos_pred = predictions[obj_mask,:]
        neg_pred = predictions[non_obj_mask,:]

        #print (obj_mask.shape)
        #print (neg_pred.shape)
        
        pos_target = targets[obj_mask]
        neg_target = targets[non_obj_mask]

        print ('#################Targets##############################')
        print (pos_target[:,:4])
        print ('#'*30)
        print ('#################Predictions##########################')
        print (pos_pred[:,:4])
        print ('#'*30)

        loss_x = self.mse_loss(pos_pred[:,0],pos_target[:,0])
        loss_y = self.mse_loss(pos_pred[:,1],pos_target[:,1]) 
        loss_h = self.mse_loss(pos_pred[:,2],pos_target[:,2]) 
        loss_w = self.mse_loss(pos_pred[:,3],pos_target[:,3])

        print (loss_x.data[0], loss_y.data[0], loss_h.data[0], loss_w.data[0])

        try :

            loss_conf_obj = self.bce_loss(pos_pred[:,4],pos_target[:,4])
            loss_conf_nonobj = 0.5*self.bce_loss(neg_pred[:,4],0*neg_target[:,4])
            #loss_conf_nonobj = 0
            loss_class = self.bce_loss(pos_pred[:,5:], pos_target[:,5:])
            print (loss_conf_obj.data[0], loss_conf_nonobj.data[0], loss_class.data[0] )

        except Exception as e:
            print (e)

        total_loss = 2.5*(loss_x + loss_y) + 2.5*(loss_w + loss_h) + (loss_conf_obj + loss_conf_nonobj + loss_class)

        return predictions,total_loss
