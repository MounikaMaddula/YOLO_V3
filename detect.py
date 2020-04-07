"""Code to detect objects in an image"""

import torch
from skimage import io
from skimage.color import gray2rgb
from skimage.transform import resize
from torchvision import transforms
from torch.autograd import Variable
import sys
import warnings
warnings.filterwarnings("ignore")

from yolo_v3 import YOLO_V3
from utils import NMS

def process_image(image, transforms = transforms.Normalize([0.43, 0.46, 0.45], [0.24, 0.23, 0.24])):

    image = io.imread(image) 
    try :
        h,w,_ = image.shape  #h,w,3
    except :
        image = gray2rgb(image)
        h,w,_ = image.shape 

    new_h,new_w = 416,416
    scale = new_h/h , new_w/w

    image = resize(image,(416,416)) #h,w,3
    image = image.transpose((2, 0, 1)) #3,h,w
    image = torch.from_numpy(image)
    image = transforms(image) #3,h,w

    return image, scale


def main(image):
    
    image, scale = process_image(image)

    model = YOLO_V3(in_channels = 3, out_classes = 2)
    #model.load_state_dict(torch.load('./bn_best_loss_chkpt.pth'))

    predictions = model(Variable(image).unsqueeze(0).float()) #1,10647,cx,cy,w,h,80
    final_coords, final_classes = NMS(predictions.data)

    final_coords[:,0] = final_coords[:,0]/scale[1]
    final_coords[:,2] = final_coords[:,2]/scale[1]
    final_coords[:,1] = final_coords[:,1]/scale[0]
    final_coords[:,3] = final_coords[:,3]/scale[0]

    print (final_classes)

if __name__ == '__main__':
    main('./Data/Images/1Spatial_Plc_-_Form_Annual_Report(Jul-02-2018)-13.jpg')