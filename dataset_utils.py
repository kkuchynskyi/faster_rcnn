import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import cv2
import xml.etree.ElementTree as ET
from PIL import Image

def get_gt_boxes(XML_IMAGE):
    tree = ET.parse(XML_IMAGE)
    root = tree.getroot()
    coordinates = []
    for member in root.findall('object'):
        coordinates.append([
            int(member[4][0].text),
            int(member[4][1].text),
            int(member[4][2].text),
            int(member[4][3].text)
            ])
    return np.array(coordinates)

def get_iterator(dir_to_data,epochs):
  df_gt_boxes = pd.read_csv(dir_to_data + '\\grayscale_train_labels.csv')
  image_names = df_gt_boxes['filename'].unique()
  all_gt_boxes = []
  images = []
  for image in image_names:
    df_one_image = df_gt_boxes[df_gt_boxes['filename']==image]
    gt_boxes = []
    for i in range(df_one_image.shape[0]):
        gt_boxes.append(df_one_image.iloc[i,4:].values)
        all_gt_boxes.append(gt_boxes)  
        img = cv2.imread(dir_to_data + "\\" + image,cv2.IMREAD_GRAYSCALE)
        images.append(np.expand_dims(np.array(img,dtype=np.float32),axis=2))
  return images,all_gt_boxes
 