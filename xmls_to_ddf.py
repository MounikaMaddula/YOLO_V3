"""Code to convert XMLs to dataframe"""

import pandas as pd 
import os
import glob
from os.path import basename
import xml.etree.ElementTree as ET
from shutil import copy

def parse_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    filename = root.find('filename').text
    for size in root.findall('size') :
        img_width, img_height,img_channel = size.find('width').text, size.find('height').text, size.find('depth').text
    labels = []
    bnd_boxes = []
    for obj in root.findall('object') :
        labels.append(obj.find('name').text)
        for bnd in obj.findall('bndbox'):
            bnd_boxes.append([bnd.find('xmin').text,bnd.find('ymin').text,bnd.find('xmax').text,bnd.find('ymax').text])
    return filename, img_width, img_height, img_channel, labels, bnd_boxes


def valid_images(data):

    data['classes'] = data['Labels'].apply(lambda x:list(set(x)))
    data['classes'] = data['classes'].apply(lambda x: [i for i in x if i not in ['table','column']])
    data['classes'] = data['classes'].apply(lambda x:len(x))
    data['Bbox'] = data['Bbox'].apply(lambda x:len(x))

    print (data[data['Bbox']==0])

    return data[data['classes']==0]

def data_folders(files):

    if not os.path.exists('Data') :
        os.mkdir('Data')

    if not os.path.exists('Data/XMLs'):
        os.mkdir('Data/XMLs')

    if not os.path.exists('Data/Images'):
        os.mkdir('Data/Images')
    
    for file in files :
        copy(file,'./Data/XMLs/{}'.format(basename(file)))
        copy('../Object_Detection/table_detection/valid_imgs/{}.jpg'.format(basename(file).replace('.xml','')),  \
            './Data/Images/{}'.format(basename(file).replace('.xml','.jpg')))


def main():

    """
    files = glob.glob('../Object_Detection/table_detection/Updated_data/*/*.xml')

    img_files = glob.glob('../Object_Detection/table_detection/valid_imgs/*.jpg')
    img_files = [basename(x) for x in img_files]

    valid_files = [x for x in files if basename(x).replace('.xml','.jpg') in img_files]

    data_folders(valid_files)
    """

    files = glob.glob('./Data/XMLs/*.xml')

    filenames = []
    img_widths = []
    img_heights = []
    img_channels = []
    labels = []
    bnd_boxes = []

    for file in files :
        filename, img_width, img_height, img_channel, label, bnd_box = parse_xml(file)
        filenames.append(filename)
        img_widths.append(img_width)
        img_heights.append(img_height)
        img_channels.append(img_channel)
        labels.append(label)
        bnd_boxes.append(bnd_box)

    data = pd.DataFrame({'Filename':filenames,'Width':img_widths,'Height':img_heights,  \
        'Channel':img_channels,'Labels':labels,'Bbox':bnd_boxes})

    print (data.head())

    data = valid_images(data)

    data.to_csv('Annotations.csv', index = False)

    print (len(data))

if __name__ == '__main__':
    main()
