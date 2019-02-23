import tensorflow as tf
import numpy as np
slim = tf.contrib.slim




def get_feature_map(img):
    """Extract feature map from input image
    Args:
      img: a 3-D tensor with shape [image_height,image_width,3]
    Returns:
      net: a 3-D tensor with shape [image_height/16,image_width/16,3]
    """
    net = slim.conv2d(img, 32, [3,3], stride=2, scope='Conv2d_0')

    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_1_depthwise')
    net = slim.conv2d(net, 64, [1,1],
                          stride=1,
                          scope='Conv2d_1_pointwise')

    net = slim.separable_conv2d(net,None, 3,
                                    stride=2,
                                    rate=1,
                                    scope='Conv2d_2_depthwise')
    net = slim.conv2d(net, 128,[1,1],
                          stride=1,
                          scope='Conv2d_2_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_3_depthwise')
    net = slim.conv2d(net, 128,[1,1],
                          stride=1,
                          scope='Conv2d_3_pointwise')
    net = slim.separable_conv2d(net,None,3,
                                    stride=2,
                                    rate=1,
                                    scope='Conv2d_4_depthwise')
    net = slim.conv2d(net, 256, [1,1],
                          stride=1,
                          scope='Conv2d_4_pointwise')

    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_5_depthwise')
    net = slim.conv2d(net, 256, [1,1],
                          stride=1,
                          scope='Conv2d_5_pointwise')
    net = slim.separable_conv2d(net,None,3,
                                    stride=2,
                                    rate=1,
                                    scope='Conv2d_6_depthwise')
    net = slim.conv2d(net, 512, [1,1],
                          stride=1,
                          scope='Conv2d_6_pointwise')

        #####
    net = slim.separable_conv2d(net,None,3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_7_depthwise')
    net = slim.conv2d(net, 512, [1, 1],
                          stride=1,
                          scope='Conv2d_7_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_8_depthwise')
    net = slim.conv2d(net, 512, [1, 1],
                          stride=1,
                          scope='Conv2d_8_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_9_depthwise')
    net = slim.conv2d(net, 512, [1, 1],
                          stride=1,
                          scope='Conv2d_9_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_10_depthwise')
    net = slim.conv2d(net, 512, [1, 1],
                          stride=1,
                          scope='Conv2d_10_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_11_depthwise')
    net = slim.conv2d(net, 512, [1, 1],
                          stride=1,
                          scope='Conv2d_11_pointwise')
        ##
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,  # 1
                                    rate=1,
                                    scope='Conv2d_12_depthwise')
    net = slim.conv2d(net, 1024, [1, 1],
                          stride=1,
                          scope='Conv2d_12_pointwise')
    net = slim.separable_conv2d(net,None, 3,
                                    stride=1,
                                    rate=1,
                                    scope='Conv2d_13_depthwise')
    net = slim.conv2d(net, 1024, [1, 1],
                          stride=1,
                          scope='Conv2d_13_pointwise')
    return net


def initialize_feature_extractor_weights(path_to_dict):
    """Initialize mobilenet v1 weights
    Returns:
      assign_op, feed_dict_init: operations 
    """
    var_arr_map = np.load(path_to_dict).item()
    # CON2d_0
    conv2d_0 = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_0/weights:0']
    conv2d_0_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/gamma:0']
    conv2d_0_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/beta:0']
    conv2d_0_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_mean:0']
    conv2d_0_bn_moving_variance = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_variance:0']
    # CON2d_1_depthwise
    conv2d_1_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/depthwise_weights:0']
    conv2d_1_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma:0']
    conv2d_1_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta:0']
    conv2d_1_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean:0']
    conv2d_1_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance:0']
    # CON2d_1_pointwise
    conv2d_1_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/weights:0']
    conv2d_1_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma:0']
    conv2d_1_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta:0']
    conv2d_1_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean:0']
    conv2d_1_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance:0']
    # CON2d_2_depthwise
    conv2d_2_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/depthwise_weights:0']
    conv2d_2_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma:0']
    conv2d_2_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta:0']
    conv2d_2_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean:0']
    conv2d_2_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance:0']
    # CON2d_2_pointwise
    conv2d_2_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/weights:0']
    conv2d_2_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma:0']
    conv2d_2_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta:0']
    conv2d_2_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean:0']
    conv2d_2_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance:0']
    # CON2d_3_depthwise
    conv2d_3_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/depthwise_weights:0']
    conv2d_3_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma:0']
    conv2d_3_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta:0']
    conv2d_3_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean:0']
    conv2d_3_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance:0']
    # CON2d_3_pointwise
    conv2d_3_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/weights:0']
    conv2d_3_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma:0']
    conv2d_3_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta:0']
    conv2d_3_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean:0']
    conv2d_3_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance:0']
    # CON2d_4_depthwise
    conv2d_4_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/depthwise_weights:0']
    conv2d_4_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma:0']
    conv2d_4_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta:0']
    conv2d_4_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean:0']
    conv2d_4_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance:0']
    # CON2d_4_pointwise
    conv2d_4_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/weights:0']
    conv2d_4_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma:0']
    conv2d_4_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta:0']
    conv2d_4_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean:0']
    conv2d_4_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance:0']
    # CON2d_5_depthwise
    conv2d_5_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/depthwise_weights:0']
    conv2d_5_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma:0']
    conv2d_5_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta:0']
    conv2d_5_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean:0']
    conv2d_5_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance:0']
    # CON2d_5_pointwise
    conv2d_5_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/weights:0']
    conv2d_5_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma:0']
    conv2d_5_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta:0']
    conv2d_5_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean:0']
    conv2d_5_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance:0']
    # CON2d_6_depthwise
    conv2d_6_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/depthwise_weights:0']
    conv2d_6_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma:0']
    conv2d_6_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta:0']
    conv2d_6_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean:0']
    conv2d_6_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance:0']
    # CON2d_6_pointwise
    conv2d_6_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/weights:0']
    conv2d_6_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma:0']
    conv2d_6_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta:0']
    conv2d_6_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean:0']
    conv2d_6_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance:0']
    ####
    # CON2d_7_depthwise
    conv2d_7_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/depthwise_weights:0']
    conv2d_7_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma:0']
    conv2d_7_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta:0']
    conv2d_7_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean:0']
    conv2d_7_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance:0']
    # CON2d_7_pointwise
    conv2d_7_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/weights:0']
    conv2d_7_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma:0']
    conv2d_7_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta:0']
    conv2d_7_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean:0']
    conv2d_7_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance:0']
    # CON2d_8_depthwise
    conv2d_8_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/depthwise_weights:0']
    conv2d_8_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma:0']
    conv2d_8_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta:0']
    conv2d_8_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean:0']
    conv2d_8_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance:0']
    # CON2d_8_pointwise
    conv2d_8_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/weights:0']
    conv2d_8_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma:0']
    conv2d_8_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta:0']
    conv2d_8_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean:0']
    conv2d_8_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance:0']
    # CON2d_9_depthwise
    conv2d_9_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/depthwise_weights:0']
    conv2d_9_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma:0']
    conv2d_9_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta:0']
    conv2d_9_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean:0']
    conv2d_9_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance:0']
    # CON2d_9_pointwise
    conv2d_9_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/weights:0']
    conv2d_9_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma:0']
    conv2d_9_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta:0']
    conv2d_9_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean:0']
    conv2d_9_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance:0']
    # CON2d_10_depthwise
    conv2d_10_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/depthwise_weights:0']
    conv2d_10_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma:0']
    conv2d_10_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta:0']
    conv2d_10_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean:0']
    conv2d_10_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance:0']
    # CON2d_10_pointwise
    conv2d_10_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/weights:0']
    conv2d_10_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma:0']
    conv2d_10_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta:0']
    conv2d_10_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean:0']
    conv2d_10_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance:0']
    # CON2d_11_depthwise
    conv2d_11_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/depthwise_weights:0']
    conv2d_11_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma:0']
    conv2d_11_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta:0']
    conv2d_11_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean:0']
    conv2d_11_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance:0']
    # CON2d_11_pointwise
    conv2d_11_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/weights:0']
    conv2d_11_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma:0']
    conv2d_11_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta:0']
    conv2d_11_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean:0']
    conv2d_11_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance:0']
    ###
    # CON2d_12_depthwise
    conv2d_12_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/depthwise_weights:0']
    conv2d_12_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:0']
    conv2d_12_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:0']
    conv2d_12_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean:0']
    conv2d_12_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance:0']
    # CON2d_12_pointwise
    conv2d_12_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/weights:0']
    conv2d_12_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:0']
    conv2d_12_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:0']
    conv2d_12_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean:0']
    conv2d_12_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance:0']
    # CON2d_13_depthwise
    conv2d_13_dw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/depthwise_weights:0']
    conv2d_13_dw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:0']
    conv2d_13_dw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:0']
    conv2d_13_dw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean:0']
    conv2d_13_dw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance:0']
    # CON2d_13_pointwise
    conv2d_13_pw = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/weights:0']
    conv2d_13_pw_bn_gamma = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:0']
    conv2d_13_pw_bn_beta = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:0']
    conv2d_13_pw_bn_moving_mean = var_arr_map['FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean:0']
    conv2d_13_pw_bn_moving_variance = var_arr_map[
        'FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance:0']
    assign_op, feed_dict_init = slim.assign_from_values({
        # Conv2d_0
        'Conv2d_0/weights': conv2d_0,
        'Conv2d_0/BatchNorm/gamma': conv2d_0_bn_gamma,
        'Conv2d_0/BatchNorm/beta': conv2d_0_bn_beta,
        'Conv2d_0/BatchNorm/moving_mean': conv2d_0_bn_moving_mean,
        'Conv2d_0/BatchNorm/moving_variance': conv2d_0_bn_moving_variance,
        # Conv2d_1_depthwise
        'Conv2d_1_depthwise/depthwise_weights': conv2d_1_dw,
        'Conv2d_1_depthwise/BatchNorm/gamma': conv2d_1_dw_bn_gamma,
        'Conv2d_1_depthwise/BatchNorm/beta': conv2d_1_dw_bn_beta,
        'Conv2d_1_depthwise/BatchNorm/moving_mean': conv2d_1_dw_bn_moving_mean,
        'Conv2d_1_depthwise/BatchNorm/moving_variance': conv2d_1_dw_bn_moving_variance,
        # Conv2d_1_pointwise
        'Conv2d_1_pointwise/weights': conv2d_1_pw,
        'Conv2d_1_pointwise/BatchNorm/gamma': conv2d_1_pw_bn_gamma,
        'Conv2d_1_pointwise/BatchNorm/beta': conv2d_1_pw_bn_beta,
        'Conv2d_1_pointwise/BatchNorm/moving_mean': conv2d_1_pw_bn_moving_mean,
        'Conv2d_1_pointwise/BatchNorm/moving_variance': conv2d_1_pw_bn_moving_variance,
        # Conv2d_2_depthwise
        'Conv2d_2_depthwise/depthwise_weights': conv2d_2_dw,
        'Conv2d_2_depthwise/BatchNorm/gamma': conv2d_2_dw_bn_gamma,
        'Conv2d_2_depthwise/BatchNorm/beta': conv2d_2_dw_bn_beta,
        'Conv2d_2_depthwise/BatchNorm/moving_mean': conv2d_2_dw_bn_moving_mean,
        'Conv2d_2_depthwise/BatchNorm/moving_variance': conv2d_2_dw_bn_moving_variance,
        # Conv2d_2_pointwise
        'Conv2d_2_pointwise/weights': conv2d_2_pw,
        'Conv2d_2_pointwise/BatchNorm/gamma': conv2d_2_pw_bn_gamma,
        'Conv2d_2_pointwise/BatchNorm/beta': conv2d_2_pw_bn_beta,
        'Conv2d_2_pointwise/BatchNorm/moving_mean': conv2d_2_pw_bn_moving_mean,
        'Conv2d_2_pointwise/BatchNorm/moving_variance': conv2d_2_pw_bn_moving_variance,
        # Conv2d_3_depthwise
        'Conv2d_3_depthwise/depthwise_weights': conv2d_3_dw,
        'Conv2d_3_depthwise/BatchNorm/gamma': conv2d_3_dw_bn_gamma,
        'Conv2d_3_depthwise/BatchNorm/beta': conv2d_3_dw_bn_beta,
        'Conv2d_3_depthwise/BatchNorm/moving_mean': conv2d_3_dw_bn_moving_mean,
        'Conv2d_3_depthwise/BatchNorm/moving_variance': conv2d_3_dw_bn_moving_variance,
        # Conv2d_3_pointwise
        'Conv2d_3_pointwise/weights': conv2d_3_pw,
        'Conv2d_3_pointwise/BatchNorm/gamma': conv2d_3_pw_bn_gamma,
        'Conv2d_3_pointwise/BatchNorm/beta': conv2d_3_pw_bn_beta,
        'Conv2d_3_pointwise/BatchNorm/moving_mean': conv2d_3_pw_bn_moving_mean,
        'Conv2d_3_pointwise/BatchNorm/moving_variance': conv2d_3_pw_bn_moving_variance,
        # Conv2d_4_depthwise
        'Conv2d_4_depthwise/depthwise_weights': conv2d_4_dw,
        'Conv2d_4_depthwise/BatchNorm/gamma': conv2d_4_dw_bn_gamma,
        'Conv2d_4_depthwise/BatchNorm/beta': conv2d_4_dw_bn_beta,
        'Conv2d_4_depthwise/BatchNorm/moving_mean': conv2d_4_dw_bn_moving_mean,
        'Conv2d_4_depthwise/BatchNorm/moving_variance': conv2d_4_dw_bn_moving_variance,
        # Conv2d_4_pointwise
        'Conv2d_4_pointwise/weights': conv2d_4_pw,
        'Conv2d_4_pointwise/BatchNorm/gamma': conv2d_4_pw_bn_gamma,
        'Conv2d_4_pointwise/BatchNorm/beta': conv2d_4_pw_bn_beta,
        'Conv2d_4_pointwise/BatchNorm/moving_mean': conv2d_4_pw_bn_moving_mean,
        'Conv2d_4_pointwise/BatchNorm/moving_variance': conv2d_4_pw_bn_moving_variance,
        # Conv2d_5_depthwise
        'Conv2d_5_depthwise/depthwise_weights': conv2d_5_dw,
        'Conv2d_5_depthwise/BatchNorm/gamma': conv2d_5_dw_bn_gamma,
        'Conv2d_5_depthwise/BatchNorm/beta': conv2d_5_dw_bn_beta,
        'Conv2d_5_depthwise/BatchNorm/moving_mean': conv2d_5_dw_bn_moving_mean,
        'Conv2d_5_depthwise/BatchNorm/moving_variance': conv2d_5_dw_bn_moving_variance,
        # Conv2d_5_pointwise
        'Conv2d_5_pointwise/weights': conv2d_5_pw,
        'Conv2d_5_pointwise/BatchNorm/gamma': conv2d_5_pw_bn_gamma,
        'Conv2d_5_pointwise/BatchNorm/beta': conv2d_5_pw_bn_beta,
        'Conv2d_5_pointwise/BatchNorm/moving_mean': conv2d_5_pw_bn_moving_mean,
        'Conv2d_5_pointwise/BatchNorm/moving_variance': conv2d_5_pw_bn_moving_variance,
        # Conv2d_6_depthwise
        'Conv2d_6_depthwise/depthwise_weights': conv2d_6_dw,
        'Conv2d_6_depthwise/BatchNorm/gamma': conv2d_6_dw_bn_gamma,
        'Conv2d_6_depthwise/BatchNorm/beta': conv2d_6_dw_bn_beta,
        'Conv2d_6_depthwise/BatchNorm/moving_mean': conv2d_6_dw_bn_moving_mean,
        'Conv2d_6_depthwise/BatchNorm/moving_variance': conv2d_6_dw_bn_moving_variance,
        # Conv2d_6_pointwise
        'Conv2d_6_pointwise/weights': conv2d_6_pw,
        'Conv2d_6_pointwise/BatchNorm/gamma': conv2d_6_pw_bn_gamma,
        'Conv2d_6_pointwise/BatchNorm/beta': conv2d_6_pw_bn_beta,
        'Conv2d_6_pointwise/BatchNorm/moving_mean': conv2d_6_pw_bn_moving_mean,
        'Conv2d_6_pointwise/BatchNorm/moving_variance': conv2d_6_pw_bn_moving_variance,
        ###
        # Conv2d_7_depthwise
        'Conv2d_7_depthwise/depthwise_weights': conv2d_7_dw,
        'Conv2d_7_depthwise/BatchNorm/gamma': conv2d_7_dw_bn_gamma,
        'Conv2d_7_depthwise/BatchNorm/beta': conv2d_7_dw_bn_beta,
        'Conv2d_7_depthwise/BatchNorm/moving_mean': conv2d_7_dw_bn_moving_mean,
        'Conv2d_7_depthwise/BatchNorm/moving_variance': conv2d_7_dw_bn_moving_variance,
        # Conv2d_7_pointwise
        'Conv2d_7_pointwise/weights': conv2d_7_pw,
        'Conv2d_7_pointwise/BatchNorm/gamma': conv2d_7_pw_bn_gamma,
        'Conv2d_7_pointwise/BatchNorm/beta': conv2d_7_pw_bn_beta,
        'Conv2d_7_pointwise/BatchNorm/moving_mean': conv2d_7_pw_bn_moving_mean,
        'Conv2d_7_pointwise/BatchNorm/moving_variance': conv2d_7_pw_bn_moving_variance,
        # Conv2d_8_depthwise
        'Conv2d_8_depthwise/depthwise_weights': conv2d_8_dw,
        'Conv2d_8_depthwise/BatchNorm/gamma': conv2d_8_dw_bn_gamma,
        'Conv2d_8_depthwise/BatchNorm/beta': conv2d_8_dw_bn_beta,
        'Conv2d_8_depthwise/BatchNorm/moving_mean': conv2d_8_dw_bn_moving_mean,
        'Conv2d_8_depthwise/BatchNorm/moving_variance': conv2d_8_dw_bn_moving_variance,
        # Conv2d_8_pointwise
        'Conv2d_8_pointwise/weights': conv2d_8_pw,
        'Conv2d_8_pointwise/BatchNorm/gamma': conv2d_8_pw_bn_gamma,
        'Conv2d_8_pointwise/BatchNorm/beta': conv2d_8_pw_bn_beta,
        'Conv2d_8_pointwise/BatchNorm/moving_mean': conv2d_8_pw_bn_moving_mean,
        'Conv2d_8_pointwise/BatchNorm/moving_variance': conv2d_8_pw_bn_moving_variance,
        # Conv2d_9_depthwise
        'Conv2d_9_depthwise/depthwise_weights': conv2d_9_dw,
        'Conv2d_9_depthwise/BatchNorm/gamma': conv2d_9_dw_bn_gamma,
        'Conv2d_9_depthwise/BatchNorm/beta': conv2d_9_dw_bn_beta,
        'Conv2d_9_depthwise/BatchNorm/moving_mean': conv2d_9_dw_bn_moving_mean,
        'Conv2d_9_depthwise/BatchNorm/moving_variance': conv2d_9_dw_bn_moving_variance,
        # Conv2d_9_pointwise
        'Conv2d_9_pointwise/weights': conv2d_9_pw,
        'Conv2d_9_pointwise/BatchNorm/gamma': conv2d_9_pw_bn_gamma,
        'Conv2d_9_pointwise/BatchNorm/beta': conv2d_9_pw_bn_beta,
        'Conv2d_9_pointwise/BatchNorm/moving_mean': conv2d_9_pw_bn_moving_mean,
        'Conv2d_9_pointwise/BatchNorm/moving_variance': conv2d_9_pw_bn_moving_variance,
        # Conv2d_10_depthwise
        'Conv2d_10_depthwise/depthwise_weights': conv2d_10_dw,
        'Conv2d_10_depthwise/BatchNorm/gamma': conv2d_10_dw_bn_gamma,
        'Conv2d_10_depthwise/BatchNorm/beta': conv2d_10_dw_bn_beta,
        'Conv2d_10_depthwise/BatchNorm/moving_mean': conv2d_10_dw_bn_moving_mean,
        'Conv2d_10_depthwise/BatchNorm/moving_variance': conv2d_10_dw_bn_moving_variance,
        # Conv2d_10_pointwise
        'Conv2d_10_pointwise/weights': conv2d_10_pw,
        'Conv2d_10_pointwise/BatchNorm/gamma': conv2d_10_pw_bn_gamma,
        'Conv2d_10_pointwise/BatchNorm/beta': conv2d_10_pw_bn_beta,
        'Conv2d_10_pointwise/BatchNorm/moving_mean': conv2d_10_pw_bn_moving_mean,
        'Conv2d_10_pointwise/BatchNorm/moving_variance': conv2d_10_pw_bn_moving_variance,
        # Conv2d_11_depthwise
        'Conv2d_11_depthwise/depthwise_weights': conv2d_11_dw,
        'Conv2d_11_depthwise/BatchNorm/gamma': conv2d_11_dw_bn_gamma,
        'Conv2d_11_depthwise/BatchNorm/beta': conv2d_11_dw_bn_beta,
        'Conv2d_11_depthwise/BatchNorm/moving_mean': conv2d_11_dw_bn_moving_mean,
        'Conv2d_11_depthwise/BatchNorm/moving_variance': conv2d_11_dw_bn_moving_variance,
        # Conv2d_11_pointwise
        'Conv2d_11_pointwise/weights': conv2d_11_pw,
        'Conv2d_11_pointwise/BatchNorm/gamma': conv2d_11_pw_bn_gamma,
        'Conv2d_11_pointwise/BatchNorm/beta': conv2d_11_pw_bn_beta,
        'Conv2d_11_pointwise/BatchNorm/moving_mean': conv2d_11_pw_bn_moving_mean,
        'Conv2d_11_pointwise/BatchNorm/moving_variance': conv2d_11_pw_bn_moving_variance,
        ###
        # Conv2d_12_depthwise
        'Conv2d_12_depthwise/depthwise_weights': conv2d_12_dw,
        'Conv2d_12_depthwise/BatchNorm/gamma': conv2d_12_dw_bn_gamma,
        'Conv2d_12_depthwise/BatchNorm/beta': conv2d_12_dw_bn_beta,
        'Conv2d_12_depthwise/BatchNorm/moving_mean': conv2d_12_dw_bn_moving_mean,
        'Conv2d_12_depthwise/BatchNorm/moving_variance': conv2d_12_dw_bn_moving_variance,
        # Conv2d_12_pointwise
        'Conv2d_12_pointwise/weights': conv2d_12_pw,
        'Conv2d_12_pointwise/BatchNorm/gamma': conv2d_12_pw_bn_gamma,
        'Conv2d_12_pointwise/BatchNorm/beta': conv2d_12_pw_bn_beta,
        'Conv2d_12_pointwise/BatchNorm/moving_mean': conv2d_12_pw_bn_moving_mean,
        'Conv2d_12_pointwise/BatchNorm/moving_variance': conv2d_12_pw_bn_moving_variance,
        # Conv2d_13_depthwise
        'Conv2d_13_depthwise/depthwise_weights': conv2d_13_dw,
        'Conv2d_13_depthwise/BatchNorm/gamma': conv2d_13_dw_bn_gamma,
        'Conv2d_13_depthwise/BatchNorm/beta': conv2d_13_dw_bn_beta,
        'Conv2d_13_depthwise/BatchNorm/moving_mean': conv2d_13_dw_bn_moving_mean,
        'Conv2d_13_depthwise/BatchNorm/moving_variance': conv2d_13_dw_bn_moving_variance,
        # Conv2d_13_pointwise
        'Conv2d_13_pointwise/weights': conv2d_13_pw,
        'Conv2d_13_pointwise/BatchNorm/gamma': conv2d_13_pw_bn_gamma,
        'Conv2d_13_pointwise/BatchNorm/beta': conv2d_13_pw_bn_beta,
        'Conv2d_13_pointwise/BatchNorm/moving_mean': conv2d_13_pw_bn_moving_mean,
        'Conv2d_13_pointwise/BatchNorm/moving_variance': conv2d_13_pw_bn_moving_variance,
    })
    return assign_op, feed_dict_init
