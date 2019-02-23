import tensorflow as tf 
from layer_utils.anchor_target import box_encoding,iou


def proposal_target_layer(rois,gt_boxes):
    """Assign targets to Fast-RCNN.
    Args:
        rois: a tensor with shape [num_rois,5]
        gt_boxes: a tensor with shape [num_gt_boxes,4],contains groundtruth boxes for input image.
    Returns:
        bbox_target_data; a tensor with shape [num_true_bboxes,4].
        labels: a tensor with shape [num_rois,].
        fg_inds: a tensor with shape[num_positive_labels,1], contains indices of rois where max_overlaps 
            with gt_boxes >= 0.5
    """
    iou_matrix = iou(rois[:,1:],gt_boxes)
    arg = tf.cast(tf.argmax(iou_matrix, axis=1), tf.int32)
    labels = tf.ones([tf.shape(arg)[0], 1], tf.int32)*-1

    max_overlaps = tf.reshape(tf.reduce_max(iou_matrix,axis=1),[-1])
    fg_inds = tf.reshape(tf.where(max_overlaps>=0.5),[-1])

    labels = tf.where((max_overlaps>=0.3)&(max_overlaps < 0.5), tf.zeros_like(labels), labels)
    labels = tf.where(max_overlaps >= 0.5, tf.ones_like(labels), labels)

    new_rois = tf.gather(rois,fg_inds)

    true_bboxes =  tf.gather(gt_boxes,tf.gather(arg,fg_inds))
    bbox_target_data = box_encoding(new_rois[:,1:], true_bboxes)
    return bbox_target_data ,tf.reshape(labels,[-1]),fg_inds