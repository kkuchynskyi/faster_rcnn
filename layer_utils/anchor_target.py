import tensorflow as tf 
import numpy as np 

def anchor_target(gt_boxes,anchors,image_width,image_height):
    """Assign targets to RPN.
    Args:
      gt_boxes: a tensor with shape [num_gt_boxes,4],contains groundtruth boxes for input image.
      anchors: a tensor with shape [num_anchors,4]
      image_height: height of input image.
      image_width: width of input image.
    Returns:
      bboxes; a tensor with shape [num_anchors,4].
      labels: a tensor with shape [num_labels,].
      indices_of_true_bboxes: a tensor with shape[num_positive_labels,1], contains indices of anchors where max_overlaps 
            with gt_boxes >= 0.5    
    """
    total_anchors = anchors.get_shape()[0]

    indices = tf.where((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (
        anchors[:, 2] < image_width) & (anchors[:, 3] < image_height))
    anchors = tf.gather_nd(anchors, indices)

    labels = tf.ones([tf.shape(indices)[0], 1], tf.int32)*-1
    iou_matrix = iou(gt_boxes, anchors)
    arg = tf.cast(tf.argmax(iou_matrix, axis=0), tf.int32)

    new_indices = tf.stack([arg, tf.range(tf.shape(indices)[0])], axis=1)
    max_overlaps = tf.gather_nd(iou_matrix, new_indices)
    labels = tf.where(max_overlaps < 0.3, tf.zeros_like(labels), labels)
    labels = tf.where(max_overlaps >= 0.5, tf.ones_like(labels), labels)
    indices_for_boxes = get_indices_for_1(labels)
    bboxes = box_encoding(anchors, tf.gather(gt_boxes, arg, axis=0))
    bboxes = tf.gather(bboxes,indices_for_boxes,axis=0)
    bboxes = tf.cond(
        tf.equal(tf.shape(bboxes)[1], 2), lambda: bboxes,
        lambda: tf.squeeze(bboxes,axis=1))
    
    labels = tf.scatter_update(tf.Variable(tf.ones(
        (total_anchors))*-1, dtype=tf.float32), indices, tf.cast(labels, tf.float32))
    indices_of_true_bboxes = get_indices_for_1(labels)
    return bboxes,labels,indices_of_true_bboxes


def iou(tb1, tb2):
    """Intersection over union
    Args:
      tb1: a tensor with shape [tb1_size,4].
      tb2: a tensor with shape [tb2_size,4].
    Return:
      iou_matrix: a tensor with shape [tb1_size,tb2_size]
    """
    x11, y11, x12, y12 = tf.split(tb1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(tb2, 4, axis=1)

    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou_matrix = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)
    return iou_matrix


def to_central_coordinates(boxes):
    """Convert coordinates to central coordinates.
    Args:
      boxes: a tensor with shape [num_boxes,4].
    Returns:
      xcenter, ycenter, width, height: tensors with
        shapes [num_boxes,1].
    """
    xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return xcenter, ycenter, width, height


def box_encoding(anchors, gt_boxes):
    """Encode anchor and gt_boxes. Described in Faster-RCNN paper.
    Args:
       anchors, gt_boxes: tensors with shape [num_anchors,4].
    Returns:
        tensor with shape [num_anchors,4].
    """
    ax, ay, aw, ah = to_central_coordinates(anchors)
    bx, by, bw, bh = to_central_coordinates(gt_boxes)
    tx = tf.div(tf.subtract(bx, ax), aw)
    ty = tf.div(tf.subtract(by, ay), ah)
    tw = tf.log(tf.div(bw, aw))
    th = tf.log(tf.div(bh, ah))

    return tf.transpose(tf.stack([tx, ty, tw, th]))


def get_indices_for_1(labels):
    """Return indices where labels==1
    Args:
      labels: a tensor with shape [num_labels,].
    Returns:
      a tensor with shape [num_indices,].
    """
    indices = tf.where(tf.equal(labels,1))
    return tf.cond(
        tf.equal(tf.shape(indices)[1], 2), lambda: tf.transpose(
            tf.nn.embedding_lookup(tf.transpose(indices), [0])),
        lambda: indices)