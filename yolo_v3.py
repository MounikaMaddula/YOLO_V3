"""Code for YOLO V3"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

#importing custom modules
from darknet_53 import DarkNet53

class YOLO_V3(nn.Module):

    def __init__(self, in_channels, out_classes):

        super(YOLO_V3,self).__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.detect_out = 3 *(4 + 1 + self.out_classes)
        self.conv11_anchors = [(10,13), (16,30), (33,23)] # (w,h) 
        self.conv8_anchors = [(30,61), (62,45), (59,119)]
        self.conv6_anchors = [(116,90), (156,198), (373,326)]
        self.darknet = DarkNet53(self.in_channels)
        self.Net()


    def Net(self):

        #out_channels = Anchors*(1+self.out_classes+4(bbox))
        self.conv11_detect = nn.Conv2d(in_channels = 5120, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0)

        self.conv11_up = nn.Upsample(scale_factor= 2, mode='bilinear')

        #out_channels = Anchors*(1+self.out_classes+4(bbox))
        self.conv8_detect = nn.Conv2d(in_channels = 5632, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0)

        self.conv8_up = nn.Upsample(scale_factor= 2, mode = 'bilinear')

        #out_channels = Anchors*(1+self.out_classes+4(bbox))
        self.conv6_detect = nn.Conv2d(in_channels = 5888, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0)
      
    def forward(self,x):

        conv6, conv8 ,conv11 = self.darknet(x) #N,256,52,52 ; N,512,26,26 ; N,5120,13,13

        conv11_detect = self.conv11_detect(conv11) #N,255,13,13
        
        conv11_up = self.conv11_up(conv11)
        conv8_combined = torch.cat((conv8, conv11_up), 1)
        conv8_detect = self.conv8_detect(conv8_combined) #N,255,26,26
        
        conv8_up = self.conv8_up(conv8_combined)
        conv6_combined = torch.cat((conv6, conv8_up), 1)
        conv6_detect = self.conv6_detect(conv6_combined) #N,255,52,52

        pred_conv11 = self.cnn_output_to_predictions(conv11_detect,self.conv11_anchors)
        pred_conv8 = self.cnn_output_to_predictions(conv8_detect, self.conv8_anchors)
        pred_conv6 = self.cnn_output_to_predictions(conv6_detect, self.conv6_anchors)

        out = torch.cat((pred_conv6,pred_conv8,pred_conv11),1)

        return out

    def prior_anchors(self) :

        conv11_anchors = [(a[0]/32,a[1]/32) for a in self.conv11_anchors]
        stride11 = 32
        grid_size11 = 13

        #Add the center offsets
        grid_len_11 = np.arange(grid_size11)
        x_offset11,y_offset11 = np.meshgrid(grid_len_11, grid_len_11)

        x_offset11 = torch.FloatTensor(x_offset11).view(-1,1)
        y_offset11 = torch.FloatTensor(y_offset11).view(-1,1)

        x_y_offset11 = torch.cat((x_offset11, y_offset11), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv11_anchors = torch.FloatTensor(conv11_anchors)
        conv11_anchors = conv11_anchors.repeat(grid_size11*grid_size11, 1).unsqueeze(0)

        priors11 = torch.zeros(1,13*13*3,self.out_classes+5)

        priors11[:,:,:2] = x_y_offset11
        priors11[:,:,2:4] = conv11_anchors
        priors11[:,:,:4] = priors11[:,:,:4]*stride11

        conv8_anchors = [(a[0]/16,a[1]/16) for a in self.conv8_anchors]
        stride8 = 16
        grid_size8 = 26

        #Add the center offsets
        grid_len_8 = np.arange(grid_size8)
        x_offset8,y_offset8 = np.meshgrid(grid_len_8, grid_len_8)

        x_offset8 = torch.FloatTensor(x_offset8).view(-1,1)
        y_offset8 = torch.FloatTensor(y_offset8).view(-1,1)

        x_y_offset8 = torch.cat((x_offset8, y_offset8), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv8_anchors = torch.FloatTensor(conv8_anchors)
        conv8_anchors = conv8_anchors.repeat(grid_size8*grid_size8, 1).unsqueeze(0)

        priors8 = torch.zeros(1,26*26*3,self.out_classes+5)

        priors8[:,:,:2] = x_y_offset8
        priors8[:,:,2:4] = conv8_anchors
        priors8[:,:,:4] = priors8[:,:,:4]*stride8

        conv6_anchors = [(a[0]/8,a[1]/8) for a in self.conv6_anchors]
        stride6 = 8
        grid_size6 = 52

        #Add the center offsets
        grid_len_6 = np.arange(grid_size6)
        x_offset6,y_offset6 = np.meshgrid(grid_len_6, grid_len_6)

        x_offset6 = torch.FloatTensor(x_offset6).view(-1,1)
        y_offset6 = torch.FloatTensor(y_offset6).view(-1,1)

        x_y_offset6 = torch.cat((x_offset6, y_offset6), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv6_anchors = torch.FloatTensor(conv6_anchors)
        conv6_anchors = conv6_anchors.repeat(grid_size6*grid_size6, 1).unsqueeze(0)

        priors6 = torch.zeros(1,52*52*3,self.out_classes+5)

        priors6[:,:,:2] = x_y_offset6
        priors6[:,:,2:4] = conv6_anchors
        priors6[:,:,:4] = priors6[:,:,:4]*stride6

        priors = torch.cat((priors8,priors6,priors11),1)
        priors = priors.squeeze(0)  #10647,85

        return priors

    def cnn_output_to_predictions(self,cnn_output,anchors, CUDA = False) :

        num_anchors = len(anchors)
        stride = 416/cnn_output.shape[2]
        batch_size,_,grid_size,_ = cnn_output.shape

        anchors = [(a[0]/stride,a[1]/stride) for a in anchors]

        cnn_output = cnn_output.view(batch_size,-1,grid_size*grid_size)
        cnn_output = cnn_output.transpose(1,2).contiguous()
        cnn_output = cnn_output.view(batch_size,grid_size*grid_size*num_anchors,-1)

        prediction = cnn_output.clone()

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.clamp(torch.sigmoid(prediction[:,:,0]),0,1)
        prediction[:,:,1] = torch.clamp(torch.sigmoid(prediction[:,:,1]),0,1)
        #prediction[:,:,4] = torch.clamp(torch.sigmoid(prediction[:,:,4]),0,1)
        

        #Add the center offsets
        grid_len = np.arange(grid_size)
        x_offset,y_offset = np.meshgrid(grid_len, grid_len)

        x_offset = torch.FloatTensor(x_offset).view(-1,1)
        y_offset = torch.FloatTensor(y_offset).view(-1,1)

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] = prediction[:,:,:2] + Variable(x_y_offset)

        #log space transform height and the width
        anchors = Variable(torch.FloatTensor(anchors))

        if CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

        #Softmax the class scores
        #prediction[:,:,5:] = torch.clamp(torch.sigmoid(prediction[:,:, 5:]),0,1)

        prediction[:,:,:4]  = prediction[:,:,:4] * stride
        #prediction = torch.where(torch.isnan(prediction), torch.zeros_like(prediction), prediction)

        return prediction #N,10647,cx,cy,w,h,80