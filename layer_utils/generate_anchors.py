import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

def generate_anchors(image_width,
                     image_height,
                     scales=[1.,0.9,0.8,1.1,1.2],
                     aspect_ratios=[1.],
                     anchor_stride=[16,16],
                     anchor_offset=[0, 0]):
    """Generates a grid of anchors at given parameters.

    num_locations = image_width/anchor_stride[0]*image_height/anchor_stride_[1]
    num_anchors = len(scales)*len(aspect_ratios)*len()
    Args:
      image_height: height of input image.
      image_width: width of input image.
      scales: a list of scales of anchors.
      aspect_ratio: a list of aspect ratios of anchors.
      anchor_stride: a pair of strides for anchors [stride for X,stride for Y].
      anchor_offset: a pair of offsets for anchors [offset for X,offset for Y]

    Returns:
      bboxes: a tensor with shape [num_locations*num_anchors,4]
    
    """
    # it was chosen to maximize number of anchors
    base_anchor_size = [0.65*image_width,0.02*image_height]

    height = tf.to_int32(tf.ceil(image_height / np.float32(anchor_stride[0])))
    width = tf.to_int32(tf.ceil(image_width / np.float32(anchor_stride[1])))
    scales_grid, aspect_ratios_grid = tf.meshgrid(scales, aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])

    ratio_sqrts = tf.sqrt(aspect_ratios_grid)
    
    heights = scales_grid / ratio_sqrts * base_anchor_size[0]
    widths = scales_grid * ratio_sqrts * base_anchor_size[1]
    # Get a grid of box centers
    y_centers = tf.to_float(tf.range(height))
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.to_float(tf.range(width))
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bboxes = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bboxes

def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: a tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)

###########################

def remove_invald_anchors(anchors,window):
  y_min, x_min, y_max, x_max = tf.split(
      value=anchors, num_or_size_splits=4, axis=1)

  win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack([0., 0., 224., 224.])
  coordinate_violations = tf.concat([
      tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
      tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
  ], 1)
  valid_indices = tf.reshape(
      tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
  new_anchors = tf.gather(anchors, valid_indices, axis=0)
  return new_anchors

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

def to_normalized(gt_boxes,height,width):
    x1,y1,x2,y2 = np.split(gt_boxes,4,axis=1)
    new_x1 = x1/width
    new_y1 = y1/height
    new_x2 = x2/width
    new_y2 = y2/height
    return np.concatenate([new_x1,new_y1,new_x2,new_y2],axis=1)

def compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret

def bbox_transform(ex_rois, gt_rois):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] 
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] 
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] 
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)
  print(targets_dx)
  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets


def iou(bboxes1, bboxes2):  
  x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
  x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
  xA = np.maximum(x11, np.transpose(x21))
  yA = np.maximum(y11, np.transpose(y21))
  xB = np.minimum(x12, np.transpose(x22))
  yB = np.minimum(y12, np.transpose(y22))
  interArea = np.maximum((xB - xA ), 0) * np.maximum((yB - yA ), 0)
  boxAArea = (x12 - x11 ) * (y12 - y11 )
  boxBArea = (x22 - x21 ) * (y22 - y21 )
  iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
  return iou




def corner_to_left_bottom_hw(anchors):
    x1,y1,x2,y2 = np.split(gt_boxes,4,axis=1)
    new_x1 = x1
    new_y1 = y1
    new_width = x2 - x1
    new_height = y2 - y1
    return np.concatenate([new_x1,new_y1,new_width,new_height],axis=1)

