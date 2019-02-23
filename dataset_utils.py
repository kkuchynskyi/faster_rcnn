import tensorflow as tf
import numpy as np
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
