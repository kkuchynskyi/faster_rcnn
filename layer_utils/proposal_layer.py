import tensorflow as tf 

def proposal_layer(rpn_cls_pred,rpn_bbox_pred,anchors,image_width,image_height):
  """Predict ROIs based on RPN predictions.

    Args:
      rpn_cls_pred: predictions of a box-classification layer of RPN. Tensor with shape
        [feature_map_width,feature_map_height,num_anchors*2].
      rpn_bbox_pred: predictions of a box-regression layer of RPN. Tensor with shape
        [feature_map_width,feature_map_height,num_anchors*4].
      anchors: a tensor with shape [feature_map_width*feature_map_height*num_anchors,4].
      image_width: a width of input image
      image_height: a height of input image
    
    Returns:
      rois: a tensor with shape [number_of_valid_rois,4]
      rois_scores:: a tensor with shape[number_of_valid_rois,1]

  """
  rpn_cls_pred = tf.reshape(
      rpn_cls_pred, [-1, 2])
  scores = tf.nn.softmax(rpn_cls_pred)[:, 1]
  # Decode of anchors coordinates and RPN bbox predictions to ROIs coordinates
  proposals = bbox_decode(anchors, rpn_bbox_pred)

  proposals,valid_indices = remove_invald_anchors(proposals, image_width, image_height)

  scores = tf.gather(scores,valid_indices)

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(
      proposals, scores, max_output_size=100, iou_threshold=0.99,score_threshold=0.2)

  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)
  return blob,scores


def remove_invald_anchors(anchors,image_width,image_height):
  """Removes anchors that (partially) fall outside an image.

    Also removes associated box encodings and objectness predictions.

    Args:
      anchors: a tensor with shape [feature_map_width*feature_map_height*num_anchors,4].
      mage_width: a width of input image
      image_height: a height of input image

  """
  y_min, x_min, y_max, x_max = tf.split(
    value=anchors, num_or_size_splits=4, axis=1)

  win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack([0., 0., image_width,image_height])
  coordinate_violations = tf.concat([
    tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
    tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
  ], 1)
  valid_indices = tf.reshape(
    tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
  new_anchors = tf.gather(anchors, valid_indices, axis=0)
  return new_anchors,valid_indices




def bbox_decode(boxes, deltas):
  """Decode of anchors coordinates and RPN bbox predictions to ROIs coordinates.

  Args:
     boxes: a tensor with shape [feature_map_width,feature_map_height,num_anchors*4].
     deltas: a tensor with shape [feature_map_width,feature_map_height,num_anchors*4].
  Reruns:
    tensor with shape [feature_map_width,feature_map_height,num_anchors*4].
  """
  boxes = tf.cast(boxes, deltas.dtype)
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
  ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  ctr_y = tf.add(boxes[:, 1], heights * 0.5)

  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]

  pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
  pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
  pred_w = tf.multiply(tf.exp(dw), widths)
  pred_h = tf.multiply(tf.exp(dh), heights)

  pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
  pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
  pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
  pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


