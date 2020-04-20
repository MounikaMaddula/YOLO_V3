"""Code for YOLO model training"""

import torch
from torch.autograd import Variable
import os
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(train_dataloader, val_dataloader, model,criteria,optimizer,exp_lr_scheduler,epochs,start_epoch,out_dir):
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

    best_val_loss = np.inf 

    for epoch in range(start_epoch,epochs):
        for p in model.parameters():
            p.requires_grad = True

        model = model.train()

        net_loss = 0
        for data in train_dataloader :
            image,bnd_boxes, lables =  data
            bnd_boxes = bnd_boxes.squeeze(0) #M,4
            lables = lables.squeeze(0) #M
            #image,bnd_boxes, lables = Variable(image.float()).to(device), Variable(bnd_boxes).to(device), Variable(lables).to(device)
            image,bnd_boxes, lables = Variable(image.float()), Variable(bnd_boxes), Variable(lables)
            predictions = model(image)
            _,loss = criteria(predictions, bnd_boxes,lables)
            net_loss += loss.data[0]
            #net_loss += loss.item()/256
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()
        exp_lr_scheduler.step()

        for param_group in optimizer.param_groups:
            print (param_group['lr'])
        """

        for p in model.parameters():
            p.requires_grad = False


        model = model.eval()

        net_val_loss = 0
        for data in val_dataloader :
            image,bnd_boxes, lables =  data
            bnd_boxes = bnd_boxes.squeeze(0) #M,4
            lables = lables.squeeze(0) #M
            #image,bnd_boxes, lables = Variable(image.float()).to(device), Variable(bnd_boxes).to(device), Variable(lables).to(device)
            image,bnd_boxes, lables = Variable(image.float()), Variable(bnd_boxes), Variable(lables)

            predictions = model(image)
            _,loss = criteria(predictions, bnd_boxes,lables)
            #net_loss += loss.data[0]/256
            net_val_loss += loss.item()/(9*bnd_boxes.shape[0])

        net_val_loss = net_val_loss/len(val_dataloader)

        """
        if net_loss < best_val_loss :
            best_val_loss = net_loss
            torch.save(model.state_dict(),'best_chkpt.pth')

        #torch.save(model.state_dict(),'{0}/epoch-{1}.pth'.format(out_dir,epoch))
        

        print ('Epoch - {0} ---------> Loss - {1}'.format(epoch, net_loss/len(train_dataloader)))
        print ('#'*30)

            
