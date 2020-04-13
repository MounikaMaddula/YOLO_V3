"""Code for YOLO V3"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import math

#importing custom modules
from darknet_53 import DarkNet53

class YOLO_V3(nn.Module):

    def __init__(self, in_channels, out_classes):

        super(YOLO_V3,self).__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.detect_out = 3 *(4 + 1 + self.out_classes)

        self.conv11_anchors = [(10,13),(16,30),(33,23)] #w,h
        self.conv8_anchors = [(30,61),( 62,45), (59,119)]
        self.conv6_anchors = [(116,90),(156,198),(373,326)]
        self.darknet = DarkNet53(self.in_channels)
        self.Net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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

        #print (conv11_detect, conv8_detect, conv6_detect)  

        conv11_detect = self.process_cnnoutput(conv11_detect) #N,13*13*3,85
        conv8_detect = self.process_cnnoutput(conv8_detect) #N,26*26*3,85
        conv6_detect = self.process_cnnoutput(conv6_detect) #N,52*52*3,85

        prediction = torch.cat((conv11_detect, conv8_detect, conv6_detect), 1)

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

        #Softmax the class scores
        prediction[:,:,5:] = torch.sigmoid(prediction[:,:, 5:])
        
        #print (prediction.shape)

        return prediction

    def process_cnnoutput(self, cnn_output):

        stride = 416//cnn_output.shape[2]
        batch_size,out_channels,grid_size,_ = cnn_output.shape
        bbox_attrs = 5 + self.out_classes
        num_anchors = out_channels//bbox_attrs

        cnn_output = cnn_output.view(batch_size,num_anchors*bbox_attrs,grid_size*grid_size) #N,255,M*M
        cnn_output = cnn_output.transpose(1,2).contiguous() #N,M*M,255
        cnn_output = cnn_output.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs) #N,M*M*3,85

        return cnn_output


    def prior_anchors(self) :

        conv11_anchors = [(a[0]/32,a[1]/32) for a in self.conv11_anchors]
        #conv11_anchors = [(np.sqrt(i),1/np.sqrt(i)) for i in self.conv11_anchors]
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
        #conv8_anchors = [(np.sqrt(i),1/np.sqrt(i)) for i in self.conv8_anchors]
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
        #conv6_anchors = [(np.sqrt(i),1/np.sqrt(i)) for i in self.conv6_anchors]
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

        priors = torch.cat((priors11,priors8,priors6),1)
        priors = priors.squeeze(0)  #10647,85 #10647,cx,cy,w,h,80

        #print (priors.shape)

        return priors
