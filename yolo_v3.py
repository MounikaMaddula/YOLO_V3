"""Code for YOLO V3"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import math

#importing custom modules
from darknet_53 import DarkNet53

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class YOLO_V3(nn.Module):

    def __init__(self, in_channels, out_classes):

        super(YOLO_V3,self).__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.detect_out = 3 *(4 + 1 + self.out_classes)
        #self.dropout = nn.Dropout2d(0.25)

        self.conv6_anchors = [(10,13),(16,30),(33,23)] #w,h
        self.conv8_anchors = [(30,61),( 62,45), (59,119)]
        self.conv11_anchors = [(116,90),(156,198),(373,326)]
        
        self.darknet = DarkNet53(self.in_channels)
        self.Net()
        #self.load_pretrained()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.darknet_load_pretrained()

    def darknet_load_pretrained(self):
        state_dict = torch.load('../darknet53_weights_pytorch.pth',map_location={'cuda:0': 'cpu'})
        model_state = self.darknet.state_dict()
        keys1 = list(state_dict.keys())
        keys2 = list(model_state.keys())
        count = 0
        for key2 in keys2 :
            if 'bias' not in key2 :
                model_state[key2] = state_dict[keys1[count]]
                count += 1
        self.darknet.load_state_dict(model_state)

    def load_pretrained(self):
        state_dict = torch.load('../official_yolov3_weights_pytorch.pth',map_location={'cuda:0': 'cpu'})
        model_state = self.Net.state_dict()
        keys1 = list(state_dict.keys())
        keys2 = list(model_state.keys())
        count = 0
        for key2 in keys2 :
            if 'bias' not in key2 :
                key1 = keys1[count]
                if state_dict[key1].shape == yolo_state[key2].shape :
                    model_state[key2] = state_dict[key1]
                count += 1
        self.Net.load_state_dict(model_state)

    def Net(self):

        #out_channels = Anchors*(1+self.out_classes+4(bbox))
        self.conv11_1 = nn.Conv2d(1024,512,1,bias = False)
        self.conv11_1_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv11_1_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_2 = nn.Conv2d(512,1024,3,padding=1,bias = False)
        self.conv11_2_bn = nn.BatchNorm2d(1024,momentum=0.9, eps=1e-5)
        self.conv11_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_3 = nn.Conv2d(1024,512,1,bias = False)
        self.conv11_3_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv11_3_act = nn.LeakyReLU(0.1, inplace = True)
        
        
        self.conv11_4 = nn.Conv2d(512,1024,3,padding=1,bias = False)
        self.conv11_4_bn = nn.BatchNorm2d(1024,momentum=0.9, eps=1e-5)
        self.conv11_4_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_5 = nn.Conv2d(1024,512,1,bias = False)
        self.conv11_5_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv11_5_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_6 = nn.Conv2d(512,1024,3,padding=1,bias = False)
        self.conv11_6_bn = nn.BatchNorm2d(1024,momentum=0.9, eps=1e-5)
        self.conv11_6_act = nn.LeakyReLU(0.1, inplace = True)
        
        
        #self.conv11_detect = nn.Conv2d(in_channels = 1024, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv11_detect = nn.Conv2d(in_channels = 1024, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0,bias = False)

        self.conv11_7 = nn.Conv2d(512,256,3,padding=1,bias = False)
        self.conv11_7_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv11_7_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_up = nn.Upsample(scale_factor=2, mode='nearest')

        #out_channels = Anchors*(1+self.out_classes+4(bbox))
        self.conv8_1 = nn.Conv2d(768,256,1,bias = False)
        self.conv8_1_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv8_1_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv8_2 = nn.Conv2d(256,512,3,padding=1,bias = False)
        self.conv8_2_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv8_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv8_3 = nn.Conv2d(512,256,1,bias = False)
        self.conv8_3_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv8_3_act = nn.LeakyReLU(0.1, inplace = True)
        
        
        self.conv8_4 = nn.Conv2d(256,512,3,padding=1,bias = False)
        self.conv8_4_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv8_4_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv8_5 = nn.Conv2d(512,256,1,bias = False)
        self.conv8_5_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv8_5_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv8_6 = nn.Conv2d(256,512,3,padding=1,bias = False)
        self.conv8_6_bn = nn.BatchNorm2d(512,momentum=0.9, eps=1e-5)
        self.conv8_6_act = nn.LeakyReLU(0.1, inplace = True)
        
        self.conv8_detect = nn.Conv2d(in_channels = 512, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0,bias = False)

        self.conv8_7 = nn.Conv2d(256,128,3,padding=1,bias = False)
        self.conv8_7_bn = nn.BatchNorm2d(128,momentum=0.9, eps=1e-5)
        self.conv8_7_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv8_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv6_1 = nn.Conv2d(384,128,1,bias = False)
        self.conv6_1_bn = nn.BatchNorm2d(128,momentum=0.9, eps=1e-5)
        self.conv6_1_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv6_2 = nn.Conv2d(128,256,3,padding=1,bias = False)
        self.conv6_2_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv6_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv6_3 = nn.Conv2d(256,128,1,bias = False)
        self.conv6_3_bn = nn.BatchNorm2d(128,momentum=0.9, eps=1e-5)
        self.conv6_3_act = nn.LeakyReLU(0.1, inplace = True)

        
        self.conv6_4 = nn.Conv2d(128,256,3,padding=1,bias = False)
        self.conv6_4_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv6_4_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv6_5 = nn.Conv2d(256,128,1,bias = False)
        self.conv6_5_bn = nn.BatchNorm2d(128,momentum=0.9, eps=1e-5)
        self.conv6_5_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv6_6 = nn.Conv2d(128,256,3,padding=1,bias = False)
        self.conv6_6_bn = nn.BatchNorm2d(256,momentum=0.9, eps=1e-5)
        self.conv6_6_act = nn.LeakyReLU(0.1, inplace = True)
        
        self.conv6_detect = nn.Conv2d(in_channels = 256, out_channels = self.detect_out,kernel_size = 1, stride=1,padding=0,bias = False)

      
    def forward(self,x):

        conv6, conv8 ,conv11 = self.darknet(x) #N,256,52,52 ; N,512,26,26 ; N,1024,13,13

        conv11_1 = self.conv11_1_act(self.conv11_1_bn(self.conv11_1(conv11)))
        conv11_2 = self.conv11_2_act(self.conv11_2_bn(self.conv11_2(conv11_1)))

        #conv11_2 = self.dropout(conv11_2)
        conv11_3 = self.conv11_3_act(self.conv11_3_bn(self.conv11_3(conv11_2)))
        conv11_4 = self.conv11_4_act(self.conv11_4_bn(self.conv11_4(conv11_3)))
        #conv11_4 = self.dropout(conv11_4)
        conv11_5 = self.conv11_5_act(self.conv11_5_bn(self.conv11_5(conv11_4)))
        conv11_6 = self.conv11_6_act(self.conv11_6_bn(self.conv11_6(conv11_5)))

        conv11_detect = self.conv11_detect(conv11_6) #N,255,13,13

        conv11_7 = self.conv11_7_act(self.conv11_7_bn(self.conv11_7(conv11_5)))
        conv11_up = self.conv11_up(conv11_7)
        
        #print (conv8.shape, conv11_up.shape)
        conv8_combined = torch.cat((conv8, conv11_up), 1)

        conv8_1 = self.conv8_1_act(self.conv8_1_bn(self.conv8_1(conv8_combined)))
        conv8_2 = self.conv8_2_act(self.conv8_2_bn(self.conv8_2(conv8_1)))
        #conv8_2 = self.dropout(conv8_2)
        conv8_3 = self.conv8_3_act(self.conv8_3_bn(self.conv8_3(conv8_2)))
        conv8_4 = self.conv8_4_act(self.conv8_4_bn(self.conv8_4(conv8_3)))
        #conv8_4 = self.dropout(conv8_4)
        conv8_5 = self.conv8_5_act(self.conv8_5_bn(self.conv8_5(conv8_4)))
        conv8_6 = self.conv8_6_act(self.conv8_6_bn(self.conv8_6(conv8_5)))

        conv8_detect = self.conv8_detect(conv8_6) #N,255,26,26

        conv8_7 = self.conv8_7_act(self.conv8_7_bn(self.conv8_7(conv8_5)))
        conv8_up = self.conv8_up(conv8_7)
        
        #print (conv6.shape, conv8_up.shape)
        conv6_combined = torch.cat((conv6, conv8_up), 1)

        conv6_1 = self.conv6_1_act(self.conv6_1_bn(self.conv6_1(conv6_combined)))
        conv6_2 = self.conv6_2_act(self.conv6_2_bn(self.conv6_2(conv6_1)))
        #conv6_2 = self.dropout(conv6_2)
        conv6_3 = self.conv6_3_act(self.conv6_3_bn(self.conv6_3(conv6_2)))
        conv6_4 = self.conv6_4_act(self.conv6_4_bn(self.conv6_4(conv6_3)))
        #conv6_4 = self.dropout(conv6_4)
        conv6_5 = self.conv6_5_act(self.conv6_5_bn(self.conv6_5(conv6_4)))
        conv6_6 = self.conv6_6_act(self.conv6_6_bn(self.conv6_6(conv6_5)))

        conv6_detect = self.conv6_detect(conv6_6) #N,255,52,52

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
        
        return prediction

    def process_cnnoutput(self, cnn_output):

        stride = 416//cnn_output.shape[2]
        batch_size,out_channels,grid_size,_ = cnn_output.shape
        bbox_attrs = 5 + self.out_classes
        num_anchors = out_channels//bbox_attrs

        #cnn_output - N,255,im_h,im_w
        cnn_output = cnn_output.permute(0,1,3,2).contiguous() #N,255,w,h
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

        x_offset11 = torch.FloatTensor(x_offset11).view(-1,1).to(device)
        y_offset11 = torch.FloatTensor(y_offset11).view(-1,1).to(device)

        x_y_offset11 = torch.cat((x_offset11, y_offset11), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv11_anchors = torch.FloatTensor(conv11_anchors).to(device)
        conv11_anchors = conv11_anchors.repeat(grid_size11*grid_size11, 1).unsqueeze(0)

        priors11 = torch.zeros(1,13*13*3,self.out_classes+5).to(device)

        priors11[:,:,:2] = x_y_offset11
        priors11[:,:,2:4] = conv11_anchors

        conv8_anchors = [(a[0]/16,a[1]/16) for a in self.conv8_anchors]
         stride8 = 16
        grid_size8 = 26

        #Add the center offsets
        grid_len_8 = np.arange(grid_size8)
        x_offset8,y_offset8 = np.meshgrid(grid_len_8, grid_len_8)

        x_offset8 = torch.FloatTensor(x_offset8).view(-1,1).to(device)
        y_offset8 = torch.FloatTensor(y_offset8).view(-1,1).to(device)

        x_y_offset8 = torch.cat((x_offset8, y_offset8), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv8_anchors = torch.FloatTensor(conv8_anchors).to(device)
        conv8_anchors = conv8_anchors.repeat(grid_size8*grid_size8, 1).unsqueeze(0)

        priors8 = torch.zeros(1,26*26*3,self.out_classes+5).to(device)

        priors8[:,:,:2] = x_y_offset8
        priors8[:,:,2:4] = conv8_anchors
        #priors8[:,:,:4] = priors8[:,:,:4]*stride8

        conv6_anchors = [(a[0]/8,a[1]/8) for a in self.conv6_anchors]
        stride6 = 8
        grid_size6 = 52

        #Add the center offsets
        grid_len_6 = np.arange(grid_size6)
        x_offset6,y_offset6 = np.meshgrid(grid_len_6, grid_len_6)

        x_offset6 = torch.FloatTensor(x_offset6).view(-1,1).to(device)
        y_offset6 = torch.FloatTensor(y_offset6).view(-1,1).to(device)

        x_y_offset6 = torch.cat((x_offset6, y_offset6), 1).repeat(1,3).view(-1,2).unsqueeze(0)

        conv6_anchors = torch.FloatTensor(conv6_anchors).to(device)
        conv6_anchors = conv6_anchors.repeat(grid_size6*grid_size6, 1).unsqueeze(0)

        priors6 = torch.zeros(1,52*52*3,self.out_classes+5).to(device)

        priors6[:,:,:2] = x_y_offset6
        priors6[:,:,2:4] = conv6_anchors

        return priors11.squeeze(0), priors8.squeeze(0), priors6.squeeze(0)

    def detect(self,predictions, pos_thres = 0.5, IoU_thres = 0.5):

        batch_size = predictions.shape[0]
        anchors = self.prior_anchors()
        strides = self.strides

        for batch in range(batch_size) :
            pred = predictions[batch,:]
            print (pred.shape)
            pred = pred.squeeze(0).data
            pred[:,0] = (pred[:,0] + anchors[:,0])*self.strides
            pred[:,1] = (pred[:,1] + anchors[:,1])*self.strides
            pred[:,2] = torch.exp(pred[:,2]) * anchors[:,2]
            pred[:,3] = torch.exp(pred[:,3]) * anchors[:,3]
            pred = pred[(pred[:,4]>=0.7).nonzero().squeeze(1)]
            print (pred[:,:4].shape)


if __name__ == '__main__':
    yolo = YOLO_V3(in_channels = 3, out_classes = 1)
    x = Variable(torch.Tensor(2,3,416,416))
    predictions = yolo(x)
