"""Code for training YOLO Model"""

import pandas as pd 
import numpy
import os
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 

#Importing custom modules
from yolo_v3 import YOLO_V3 
from model_loss import MultiBox_Loss
from datasets import Data_Loader
from train import train_model
from utils import *

#Defining arguments for model training
#parser = argparse.ArgumentParser()
#parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.00005')

if __name__ == '__main__':
    
    dataset = Data_Loader(img_path = './Data/Images',xml_path = './Data/XMLs')
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True, num_workers=1)

    model = YOLO_V3(in_channels = 3, out_classes = 2)
    criteria = MultiBox_Loss(yolo_model = model,pos_iou_threshold = 0.7,neg_iou_threshold = 0.3,n_sample = 256 ,pos_ratio = 0.5)

    optimizer = optim.Adam(model.parameters(), lr= 0.0001,betas=(0.5, 0.999))

    train_model(dataloader, model,criteria,optimizer,epochs = 100,start_epoch = 0,out_dir = './chkpts')