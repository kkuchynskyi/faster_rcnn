import tensorflow as tf 


def true_rcnn_cls_fn(rcnn_labels,cls_score):
    indices_for_rсnn_labels = tf.where(tf.not_equal(rcnn_labels,-1))
    cls_rcnn_labels = tf.cast(tf.gather(rcnn_labels,indices_for_rсnn_labels),tf.int32)
    cls_rcnn_predictions = tf.reshape(tf.gather(cls_score,indices_for_rсnn_labels),[-1,2])
    rcnn_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=cls_rcnn_predictions, labels=cls_rcnn_labels))
    return rcnn_cross_entropy

def true_rcnn_bbox_fn(bbox_pred,rcnn_indices_true_bboxes,rcnn_bbox_target):
    rcnn_bbox_pred = bbox_pred[:,4:]
    rcnn_bbox_pred = tf.gather(rcnn_bbox_pred,rcnn_indices_true_bboxes,axis=0)
    rcnn_bbox_loss = smoothL1(rcnn_bbox_pred,rcnn_bbox_target)
    return rcnn_bbox_loss

def smoothL1(bbox_pred,bbox_target):
    sigma_2 = 1 ** 2
    box_diff = bbox_pred - bbox_target
    abs_box_diff = tf.abs(box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss_box = tf.reduce_mean(tf.reduce_sum(
          in_loss_box,
          axis=[1]
        ))
    return loss_box