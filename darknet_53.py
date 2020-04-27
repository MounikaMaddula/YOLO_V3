"""Code for DarkNet-53"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class DarkNet53(nn.Module):

    def __init__(self, in_channels):
        
        super(DarkNet53,self).__init__()

        self.in_channels = in_channels
        self.Net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def Net(self):

        self.dropout = nn.Dropout2d(0.25)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32,kernel_size = 3, stride=1,padding=1, bias = False)
        self.conv1_bn = nn.BatchNorm2d(32,momentum=0.9, eps=1e-5)
        self.conv1_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,kernel_size = 3, stride=2,padding=1,bias = False)
        self.conv2_bn = nn.BatchNorm2d(64,momentum=0.9, eps=1e-5)
        self.conv2_act = nn.LeakyReLU(0.1, inplace = True)  

        self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 32,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv3_1_bn = nn.BatchNorm2d(32,momentum=0.9, eps=1e-5)
        self.conv3_1_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv3_2 = nn.Conv2d(in_channels = 32, out_channels = 64,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv3_2_bn = nn.BatchNorm2d(64,momentum=0.9, eps=1e-5)
        self.conv3_2_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = 3, stride=2,padding=1,bias = False)
        self.conv4_bn = nn.BatchNorm2d(128,momentum=0.9, eps=1e-5)
        self.conv4_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv5_1_1 = nn.Conv2d(in_channels = 128, out_channels = 64,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv5_1_1_bn = nn.BatchNorm2d(64,momentum=0.9, eps=1e-5)
        self.conv5_1_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv5_1_2 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv5_1_2_bn = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
        self.conv5_1_2_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv5_2_1 = nn.Conv2d(in_channels = 128, out_channels = 64,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv5_2_1_bn = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)
        self.conv5_2_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv5_2_2 = nn.Conv2d(in_channels = 64, out_channels = 128,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv5_2_2_bn = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
        self.conv5_2_2_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=2,padding=1,bias = False)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_act = nn.LeakyReLU(0.1, inplace = True)
        
        self.conv7_1_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_1_1_bn = nn.BatchNorm2d(128)
        self.conv7_1_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_1_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_1_2_bn = nn.BatchNorm2d(256)
        self.conv7_1_2_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv7_2_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_2_1_bn = nn.BatchNorm2d(128)
        self.conv7_2_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_2_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_2_2_bn = nn.BatchNorm2d(256)
        self.conv7_2_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_3_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_3_1_bn = nn.BatchNorm2d(128)
        self.conv7_3_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_3_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_3_2_bn = nn.BatchNorm2d(256)
        self.conv7_3_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_4_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_4_1_bn = nn.BatchNorm2d(128)
        self.conv7_4_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_4_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_4_2_bn = nn.BatchNorm2d(256)
        self.conv7_4_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_5_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_5_1_bn = nn.BatchNorm2d(128)
        self.conv7_5_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_5_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_5_2_bn = nn.BatchNorm2d(256)
        self.conv7_5_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_6_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_6_1_bn = nn.BatchNorm2d(128)
        self.conv7_6_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_6_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_6_2_bn = nn.BatchNorm2d(256)
        self.conv7_6_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_7_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_7_1_bn = nn.BatchNorm2d(128)
        self.conv7_7_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_7_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_7_2_bn = nn.BatchNorm2d(256)
        self.conv7_7_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv7_8_1 = nn.Conv2d(in_channels = 256, out_channels = 128,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv7_8_1_bn = nn.BatchNorm2d(128)
        self.conv7_8_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv7_8_2 = nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv7_8_2_bn = nn.BatchNorm2d(256)
        self.conv7_8_2_act = nn.LeakyReLU(0.1, inplace = True)
        
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=2,padding=1,bias = False)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv8_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_1_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_1_1_bn = nn.BatchNorm2d(256)
        self.conv9_1_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_1_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_1_2_bn = nn.BatchNorm2d(512)
        self.conv9_1_2_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv9_2_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_2_1_bn = nn.BatchNorm2d(256)
        self.conv9_2_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_2_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_2_2_bn = nn.BatchNorm2d(512)
        self.conv9_2_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_3_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_3_1_bn = nn.BatchNorm2d(256)
        self.conv9_3_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_3_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_3_2_bn = nn.BatchNorm2d(512)
        self.conv9_3_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_4_1 = nn.Conv2d(in_channels =512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_4_1_bn = nn.BatchNorm2d(256)
        self.conv9_4_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_4_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_4_2_bn = nn.BatchNorm2d(512)
        self.conv9_4_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_5_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_5_1_bn = nn.BatchNorm2d(256)
        self.conv9_5_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_5_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_5_2_bn = nn.BatchNorm2d(512)
        self.conv9_5_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_6_1 = nn.Conv2d(in_channels =512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_6_1_bn = nn.BatchNorm2d(256)
        self.conv9_6_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_6_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_6_2_bn = nn.BatchNorm2d(512)
        self.conv9_6_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_7_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_7_1_bn = nn.BatchNorm2d(256)
        self.conv9_7_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_7_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_7_2_bn = nn.BatchNorm2d(512)
        self.conv9_7_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv9_8_1 = nn.Conv2d(in_channels = 512, out_channels = 256,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv9_8_1_bn = nn.BatchNorm2d(256)
        self.conv9_8_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv9_8_2 = nn.Conv2d(in_channels = 256, out_channels = 512,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv9_8_2_bn = nn.BatchNorm2d(512)
        self.conv9_8_2_act = nn.LeakyReLU(0.1, inplace = True)
        
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 1024,kernel_size = 3, stride=2,padding=1,bias = False)
        self.conv10_bn = nn.BatchNorm2d(1024)
        self.conv10_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_1_1 = nn.Conv2d(in_channels = 1024, out_channels = 512,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv11_1_1_bn = nn.BatchNorm2d(512)
        self.conv11_1_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv11_1_2 = nn.Conv2d(in_channels = 512, out_channels = 1024,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv11_1_2_bn = nn.BatchNorm2d(1024)
        self.conv11_1_2_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv11_2_1 = nn.Conv2d(in_channels = 1024, out_channels = 512,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv11_2_1_bn = nn.BatchNorm2d(512)
        self.conv11_2_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv11_2_2 = nn.Conv2d(in_channels = 512, out_channels = 1024,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv11_2_2_bn = nn.BatchNorm2d(1024)
        self.conv11_2_2_act = nn.LeakyReLU(0.1, inplace = True) 

        self.conv11_3_1 = nn.Conv2d(in_channels = 1024, out_channels = 512,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv11_3_1_bn = nn.BatchNorm2d(512)
        self.conv11_3_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv11_3_2 = nn.Conv2d(in_channels = 512, out_channels = 1024,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv11_3_2_bn = nn.BatchNorm2d(1024)
        self.conv11_3_2_act = nn.LeakyReLU(0.1, inplace = True)

        self.conv11_4_1 = nn.Conv2d(in_channels = 1024, out_channels = 512,kernel_size = 1, stride=1,padding=0,bias = False)
        self.conv11_4_1_bn = nn.BatchNorm2d(512)
        self.conv11_4_1_act = nn.LeakyReLU(0.1, inplace = True) 
        
        self.conv11_4_2 = nn.Conv2d(in_channels = 512, out_channels = 1024,kernel_size = 3, stride=1,padding=1,bias = False)
        self.conv11_4_2_bn = nn.BatchNorm2d(1024)
        self.conv11_4_2_act = nn.LeakyReLU(0.1, inplace = True)       

    def forward(self,x):

        conv1 = self.conv1_act(self.conv1_bn(self.conv1(x)))
        conv2 = self.conv2_act(self.conv2_bn(self.conv2(conv1)))
        
        conv3_1 = self.conv3_1_act(self.conv3_1_bn(self.conv3_1(conv2)))
        conv3_2 = self.conv3_2_act(self.conv3_2_bn(self.conv3_2(conv3_1)))
        ##print (conv3_2.shape, conv2.shape)
        conv3_res = conv3_2 + conv2
        
        conv4 = self.conv4_act(self.conv4_bn(self.conv4(conv3_res)))
        ##print (conv4.shape)
        
        conv5_1_1 = self.conv5_1_1_act(self.conv5_1_1_bn(self.conv5_1_1(conv4)))
        conv5_1_2 = self.conv5_1_2_act(self.conv5_1_2_bn(self.conv5_1_2(conv5_1_1)))
        conv5_1_res = conv5_1_2 + conv4
        
        conv5_2_1 = self.conv5_2_1_act(self.conv5_2_1_bn(self.conv5_2_1(conv5_1_res)))
        conv5_2_2 = self.conv5_2_2_act(self.conv5_2_2_bn(self.conv5_2_2(conv5_2_1)))
        #print (conv5_2_2.shape, conv5_1_res.shape)
        conv5_2_res = conv5_2_2 + conv5_1_res 
        
        conv6 = self.conv6_act(self.conv6_bn(self.conv6(conv5_2_res)))
        ##print (conv6.shape)

        conv7_1_1 = self.conv7_1_1_act(self.conv7_1_1_bn(self.conv7_1_1(conv6)))
        conv7_1_2 = self.conv7_1_2_act(self.conv7_1_2_bn(self.conv7_1_2(conv7_1_1)))
        #conv7_1_2 = self.dropout(conv7_1_2)
        #print (conv7_1_2.shape, conv6.shape)
        conv7_1_res = conv7_1_2 + conv6
        
        ##print (conv7_1_res.shape)

        conv7_2_1 = self.conv7_2_1_act(self.conv7_2_1_bn(self.conv7_2_1(conv7_1_res)))
        conv7_2_2 = self.conv7_2_2_act(self.conv7_2_2_bn(self.conv7_2_2(conv7_2_1)))
        #conv7_2_2 = self.dropout(conv7_2_2)
        #print (conv7_2_2.shape, conv7_1_res.shape)
        conv7_2_res = conv7_2_2 + conv7_1_res

        conv7_3_1 = self.conv7_3_1_act(self.conv7_3_1_bn(self.conv7_3_1(conv7_2_res)))
        conv7_3_2 = self.conv7_3_2_act(self.conv7_3_2_bn(self.conv7_3_2(conv7_3_1)))
        conv7_3_2 = self.dropout(conv7_3_2)
        #print (conv7_1_2.shape, conv7_2_res.shape)
        conv7_3_res = conv7_3_2 + conv7_2_res

        conv7_4_1 = self.conv7_4_1_act(self.conv7_4_1_bn(self.conv7_4_1(conv7_3_res)))
        conv7_4_2 = self.conv7_4_2_act(self.conv7_4_2_bn(self.conv7_4_2(conv7_4_1)))
        #conv7_4_2 = self.dropout(conv7_4_2)
        #print (conv7_4_2.shape, conv7_3_res.shape)
        conv7_4_res = conv7_4_2 + conv7_3_res

        conv7_5_1 = self.conv7_5_1_act(self.conv7_5_1_bn(self.conv7_5_1(conv7_4_res)))
        conv7_5_2 = self.conv7_5_2_act(self.conv7_5_2_bn(self.conv7_5_2(conv7_5_1)))
        conv7_5_2 = self.dropout(conv7_5_2)
        #print (conv7_5_2.shape, conv7_4_res.shape)
        conv7_5_res = conv7_5_2 + conv7_4_res
        
        conv7_6_1 = self.conv7_6_1_act(self.conv7_6_1_bn(self.conv7_6_1(conv7_5_res)))
        conv7_6_2 = self.conv7_6_2_act(self.conv7_6_2_bn(self.conv7_6_2(conv7_6_1)))
        #conv7_6_2 = self.dropout(conv7_6_2)
        #print (conv7_6_2.shape, conv7_5_res.shape)
        conv7_6_res = conv7_6_2 + conv7_5_res

        conv7_7_1 = self.conv7_7_1_act(self.conv7_7_1_bn(self.conv7_7_1(conv7_6_res)))
        conv7_7_2 = self.conv7_7_2_act(self.conv7_7_2_bn(self.conv7_7_2(conv7_7_1)))
        conv7_7_2 = self.dropout(conv7_7_2)
        #print (conv7_7_2.shape, conv7_6_res.shape)
        conv7_7_res = conv7_7_2 + conv7_6_res

        conv7_8_1 = self.conv7_8_1_act(self.conv7_8_1_bn(self.conv7_8_1(conv7_7_res)))
        conv7_8_2 = self.conv7_8_2_act(self.conv7_8_2_bn(self.conv7_8_2(conv7_8_1)))
        #conv7_8_2 = self.dropout(conv7_8_2)
        #print (conv7_8_2.shape, conv7_7_res.shape)
        conv7_8_res = conv7_8_2 + conv7_7_res
        
        conv8 = self.conv8_act(self.conv8_bn(self.conv8(conv7_8_res)))
        #conv8 = self.conv8_act(self.conv8_bn(self.conv8(conv7_2_res)))
        #print (conv8.shape)

        conv9_1_1 = self.conv9_1_1_act(self.conv9_1_1_bn(self.conv9_1_1(conv8)))
        conv9_1_2 = self.conv9_1_2_act(self.conv9_1_2_bn(self.conv9_1_2(conv9_1_1)))
        #conv9_1_2 = self.dropout(conv9_1_2)
        #print (conv9_1_2.shape, conv8.shape)
        conv9_1_res = conv9_1_2 + conv8
        
        #print (conv9_1_res.shape)

        conv9_2_1 = self.conv9_2_1_act(self.conv9_2_1_bn(self.conv9_2_1(conv9_1_res)))
        conv9_2_2 = self.conv9_2_2_act(self.conv9_2_2_bn(self.conv9_2_2(conv9_2_1)))
        #conv9_2_2 = self.dropout(conv9_2_2)
        #print (conv9_2_2.shape, conv9_1_res.shape)
        conv9_2_res = conv9_2_2 + conv9_1_res
        
        conv9_3_1 = self.conv9_3_1_act(self.conv9_3_1_bn(self.conv9_3_1(conv9_2_res)))
        conv9_3_2 = self.conv9_3_2_act(self.conv9_3_2_bn(self.conv9_3_2(conv9_3_1)))
        conv9_3_2 = self.dropout(conv9_3_2)
        #print (conv9_3_2.shape, conv9_2_res.shape)
        conv9_3_res = conv9_3_2 + conv9_2_res

        conv9_4_1 = self.conv9_4_1_act(self.conv9_4_1_bn(self.conv9_4_1(conv9_3_res)))
        conv9_4_2 = self.conv9_4_2_act(self.conv9_4_2_bn(self.conv9_4_2(conv9_4_1)))
        #conv9_4_2 = self.dropout(conv9_4_2)
        #print (conv9_4_2.shape, conv9_3_res.shape)
        conv9_4_res = conv9_4_2 + conv9_3_res
        
        conv9_5_1 = self.conv9_5_1_act(self.conv9_5_1_bn(self.conv9_5_1(conv9_4_res)))
        conv9_5_2 = self.conv9_5_2_act(self.conv9_5_2_bn(self.conv9_5_2(conv9_5_1)))
        conv9_5_2 = self.dropout(conv9_5_2)
        #print (conv9_5_2.shape, conv9_4_res.shape)
        conv9_5_res = conv9_5_2 + conv9_4_res
        
        conv9_6_1 = self.conv9_6_1_act(self.conv9_6_1_bn(self.conv9_6_1(conv9_5_res)))
        conv9_6_2 = self.conv9_6_2_act(self.conv9_6_2_bn(self.conv9_6_2(conv9_6_1)))
        #conv9_6_2 = self.dropout(conv9_6_2)
        #print (conv9_6_2.shape, conv9_5_res.shape)
        conv9_6_res = conv9_6_2 + conv9_5_res

        conv9_7_1 = self.conv9_7_1_act(self.conv9_7_1_bn(self.conv9_7_1(conv9_6_res)))
        conv9_7_2 = self.conv9_7_2_act(self.conv9_7_2_bn(self.conv9_7_2(conv9_7_1)))
        conv9_7_2 = self.dropout(conv9_7_2)
        #print (conv9_7_2.shape, conv9_6_res.shape)
        conv9_7_res = conv9_7_2 + conv9_6_res

        conv9_8_1 = self.conv9_8_1_act(self.conv9_8_1_bn(self.conv9_8_1(conv9_7_res)))
        conv9_8_2 = self.conv9_8_2_act(self.conv9_8_2_bn(self.conv9_8_2(conv9_8_1)))
        #conv9_8_2 = self.dropout(conv9_8_2)
        #print (conv9_8_2.shape, conv9_7_res.shape)
        conv9_8_res = conv9_8_2 + conv9_7_res
        
        conv10 = self.conv10_act(self.conv10_bn(self.conv10(conv9_8_res)))
        #conv10 = self.conv10_act(self.conv10_bn(self.conv10(conv9_2_res)))
        ##print (conv10.shape)

        conv11_1_1 = self.conv11_1_1_act(self.conv11_1_1_bn(self.conv11_1_1(conv10)))
        conv11_1_2 = self.conv11_1_2_act(self.conv11_1_2_bn(self.conv11_1_2(conv11_1_1)))
        #conv11_1_2 = self.dropout(conv11_1_2)
        #print (conv11_1_2.shape, conv10.shape)
        conv11_1_res = conv11_1_2 + conv10
        
        conv11_2_1 = self.conv11_2_1_act(self.conv11_2_1_bn(self.conv11_2_1(conv11_1_res)))
        conv11_2_2 = self.conv11_2_2_act(self.conv11_2_2_bn(self.conv11_2_2(conv11_2_1)))
        conv11_2_2 = self.dropout(conv11_2_2)
        #print (conv11_1_2.shape, conv11_1_res.shape)
        conv11_2_res = conv11_2_2 + conv11_1_res
        
        conv11_3_1 = self.conv11_3_1_act(self.conv11_3_1_bn(self.conv11_3_1(conv11_2_res)))
        conv11_3_2 = self.conv11_3_2_act(self.conv11_3_2_bn(self.conv11_3_2(conv11_3_1)))
        #conv11_3_2 = self.dropout(conv11_3_2)
        #print (conv11_3_2.shape, conv11_2_res.shape)
        conv11_3_res = conv11_3_2 + conv11_2_res

        conv11_4_1 = self.conv11_4_1_act(self.conv11_4_1_bn(self.conv11_4_1(conv11_3_res)))
        conv11_4_2 = self.conv11_4_2_act(self.conv11_4_2_bn(self.conv11_4_2(conv11_4_1)))
        #conv11_4_2 = self.dropout(conv11_4_2)
        #print (conv11_4_2.shape, conv11_3_res.shape)
        conv11_4_res = conv11_4_2 + conv11_3_res
        
        
        return conv6, conv8 ,conv11_4_res
