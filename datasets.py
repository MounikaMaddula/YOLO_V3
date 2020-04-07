"""Code for dataloading"""

import pandas as pd 
import torch
import os
from os.path import basename
import glob
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms
from torch.autograd import Variable
from skimage import io
from skimage.color import gray2rgb
from skimage.transform import resize
import xml.etree.ElementTree as ET
import numpy as np 
import warnings
warnings.filterwarnings("ignore")

from utils import corners_to_cxcy

class Data_Loader(Dataset):

    def __init__(self, img_path,xml_path,out_size = (416,416)):

        super(Data_Loader,self).__init__()

        self.img_path = img_path
        self.xml_path = xml_path
        self.out_size = out_size
        self.transforms = transforms.Normalize([0.43, 0.46, 0.45], [0.24, 0.23, 0.24])
        self._lables_dict()

        self.images = glob.glob(self.img_path+'/*.jpg')

    def _lables_dict(self):

        lables_dict = {}
        lables_dict['table'] = 0
        lables_dict['column'] = 1

        self.lables_dict = lables_dict

    def _resize(self,img, img_bboxes):

        image = io.imread(img) 
        try :
            h,w,_ = image.shape  #h,w,3
        except :
            image = gray2rgb(image)
            h,w,_ = image.shape 

        new_h,new_w = self.out_size
        scale = new_h/h , new_w/w

        image = resize(image,self.out_size) #h,w,3
        image = image.transpose((2, 0, 1)) #3,h,w
        image = torch.from_numpy(image)
        image = self.transforms(image) #3,h,w

        img_bboxes[:,0] = img_bboxes[:,0] * scale[1]
        img_bboxes[:,1] = img_bboxes[:,1]*scale[0]
        img_bboxes[:,2] = img_bboxes[:,2] * scale[1]
        img_bboxes[:,3] =  img_bboxes[:,3]*scale[0]

        #img_bboxes = img_bboxes.unsqueeze(0) #1,M,4

        return image, img_bboxes


    def _parse_xml(self,filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        filename = root.find('filename').text
        for size in root.findall('size') :
            img_width, img_height,img_channel = size.find('width').text, size.find('height').text, size.find('depth').text
        lables = []
        bnd_boxes = []
        for obj in root.findall('object') :
            lables.append(obj.find('name').text)
            for bnd in obj.findall('bndbox'):
                bnd_boxes.append([float(bnd.find('xmin').text),float(bnd.find('ymin').text),  \
                    float(bnd.find('xmax').text),float(bnd.find('ymax').text)])
        return img_width, img_height, img_channel, lables, bnd_boxes


    def __getitem__(self,ix):

        image = self.images[ix]
        #print (image)
        xml_file = self.xml_path+'/' + basename(image).replace('.jpg','.xml')

        img_width, img_height, img_channel, lables, bnd_boxes = self._parse_xml(xml_file)

        bnd_boxes = torch.from_numpy(np.asarray(bnd_boxes)) #M,4
        #print (bnd_boxes.shape)
        bnd_boxes = corners_to_cxcy(bnd_boxes) #M,4

        lables = [self.lables_dict[x] for x in lables]
        lables = torch.from_numpy(np.asarray(lables)) #M

        image,bnd_boxes = self._resize(image, bnd_boxes) #3,h,w;M,4

        return image,bnd_boxes, lables


    def __len__(self):
        return len(self.images)

def main():

    check = Data_Loader(img_path = './Data/Images',xml_path = './Data/XMLs')

    dataloader =  DataLoader(check,batch_size=1,shuffle=True, num_workers=4)
    #c = 0
    for i in dataloader:
        try :
            image,bnd_boxes, lables =  i
            print (image.shape, bnd_boxes.shape, lables.shape)
        except Exception as e :
            print (e)
            print ('##############################################')
            exit()


if __name__ == '__main__':
    main()