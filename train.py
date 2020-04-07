"""Code for YOLO model training"""

import torch
from torch.autograd import Variable
import os
import numpy as np

def train_model(dataloader, model,criteria,optimizer,epochs,start_epoch,out_dir):
    """
        Model training function

        Arguments:
            model - yolo model
            criteria - criteria for calculating loss
            optimizer - optimizer for backpropogation
            epochs - no. of epochs needed
            start_epoch - start epoch for model training
            out_dir - directory for saving checkpoints
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    best_loss = np.inf 

    for epoch in range(start_epoch,epochs):
        net_loss = 0
        for data in dataloader :
            image,bnd_boxes, lables =  data
            bnd_boxes = bnd_boxes.squeeze(0) #M,4
            lables = lables.squeeze(0) #M
            try :
                image,bnd_boxes, lables = Variable(image.float()), Variable(bnd_boxes), Variable(lables)
                predictions = model(image)
                _,loss = criteria(predictions, bnd_boxes,lables)
                net_loss += loss.data[0]/256
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                optimizer.step()
            except Exception as e :
                print (e)
                print (image.shape, bnd_boxes.shape, lables.shape)

        print ('Epoch - {0} ---------> Loss - {1}'.format(epoch, net_loss))
        print ('#'*30)

        torch.save(model.state_dict(),'{0}/epoch-{1}.pth'.format(out_dir,epoch))