import tensorflow as tf 
import numpy as np

slim = tf.contrib.slim

def roi_pool_layer(net,rois):
    """RoI pooling layer uses max pooling to convert the
    features inside any valid region of interest into [7,7] feature map.

    Args:
        net: a tensor with shape [feature_map_height,feature_map_width,1024], it is an output of mobilenet v1
        rois: a tensor with shape [num_rois,5]
    Returns:
        fc: a tensor with shape [num_rois,7,7,1024]    
    """
    net_shape = tf.shape(net)
    height = (tf.to_float(net_shape[0]) - 1.) * 16.
    width = (tf.to_float(net_shape[1]) - 1.) * 16.
    batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
    x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
    y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
    x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
    y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
    
    # Won't be back-propagated to rois anyway, but to save time
    bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
    pre_pool_size = 7 * 2
    crops = tf.image.crop_and_resize(net, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    fc = slim.max_pool2d(crops, [2, 2], padding='SAME')
    return fc