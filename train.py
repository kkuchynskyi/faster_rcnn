import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import itertools
from layer_utils.generate_anchors import *
from PIL import Image
from layer_utils.feature_map import get_feature_map,initialize_feature_extractor_weights
from layer_utils.anchor_target import *
from layer_utils.proposal_layer import *
from layer_utils.roi_pool_layer import roi_pool_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from loss_utils import *

slim = tf.contrib.slim
print(tf.VERSION)   
img_to_feed = tf.placeholder(shape=(1024,1024,1),dtype=tf.float32)
img_to_feed = tf.expand_dims(tf.image.per_image_standardization(img_to_feed),0)
gt_boxes_to_feed = tf.placeholder(shape=(None,4),dtype=tf.float32)


NUM_ANCHORS = 5
batch_norm_params = {
    'center': True,
    'scale': True,
    'decay': 0.9997,
    'epsilon': 0.001,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
}
initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                    activation_fn=tf.nn.relu6,
                    normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        net = get_feature_map(img_to_feed)
        ###Finish of feature extractor. Start RPN  ###

        # small network over the convolutional feature map,n=3
        rpn = slim.conv2d(net, 512, [3, 3], trainable=True, weights_initializer=initializer,
                          scope="rpn_conv/3x3")

        rpn_cls_score = slim.conv2d(rpn, NUM_ANCHORS * 2, [1, 1], trainable=True,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')

        class_predictions_with_background = tf.reshape(rpn_cls_score, [-1, 2])


        rpn_bbox_pred = slim.conv2d(net, NUM_ANCHORS*4, [1, 1], trainable=True,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

anchors = generate_anchors(1024,1024)

rois,rois_score = proposal_layer(rpn_cls_score,rpn_bbox_pred,anchors,1024,1024)

fc7 = roi_pool_layer(net,rois)

fc7_ = tf.reduce_mean(fc7, axis=[1, 2])

cls_score = slim.fully_connected(fc7_, 2,
                                 weights_initializer=initializer,
                                 trainable=True,
                                 activation_fn=None, scope='cls_score2')


bbox_pred = slim.fully_connected(fc7_,2 * 4,
                                 weights_initializer=initializer_bbox,
                                 trainable=True,
                                 activation_fn=None, scope='bbox_pred2')

### LOSS
image_height = 1024
image_width = 1024


rpn_bboxes_target, rpn_labels, rpn_indices_true_bboxes = anchor_target(
    gt_boxes_to_feed, anchors, image_width, image_height)

rcnn_bbox_target, rcnn_labels, rcnn_indices_true_bboxes = proposal_target_layer(rois, gt_boxes_to_feed)
# RPN LOSS
indices_for_rpn_labels = tf.where(tf.not_equal(rpn_labels,-1))
rpn_labels = tf.cast(tf.gather(rpn_labels,indices_for_rpn_labels),tf.int32)
rpn_predictions_cls = tf.reshape(tf.gather(class_predictions_with_background,indices_for_rpn_labels),[-1,2])

rpn_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=rpn_predictions_cls, labels=rpn_labels))

rpn_bbox = tf.gather(rpn_bbox_pred,rpn_indices_true_bboxes,axis=0)
rpn_bbox_loss = smoothL1(rpn_bbox,rpn_bboxes_target)
# RCNN LOSS
cond = tf.equal(tf.size(rcnn_indices_true_bboxes), 0)
rcnn_cross_entropy = tf.cond(cond,lambda:0.,lambda:true_rcnn_cls_fn(rcnn_labels,cls_score))
rcnn_bbox_loss = tf.cond(cond,lambda:0.,lambda:true_rcnn_bbox_fn(bbox_pred,rcnn_indices_true_bboxes,rcnn_bbox_target))

### TOTAL LOSS
# total_loss = rpn_cross_entropy+rpn_bbox_loss+rcnn_bbox_loss+rcnn_cross_entropy
# total_loss = rpn_bbox_loss+rcnn_bbox_loss
# total_loss = rpn_cross_entropy
# total_loss = rpn_cross_entropy
# total_loss = rpn_cross_entropy+rcnn_cross_entropy
# total_loss = rpn_cross_entropy+rcnn_cross_entropy
# total_loss = rpn_cross_entropy+rpn_bbox_loss
total_loss = rpn_cross_entropy+rpn_bbox_loss+rcnn_bbox_loss



lr = tf.Variable(0.001, trainable=False)
optimizer = tf.train.MomentumOptimizer(lr, 0.9)
gvs = optimizer.compute_gradients(total_loss)
train_op = optimizer.apply_gradients(gvs)


df = pd.read_csv('/faster_rcnn_data/grayscale_train/grayscale_train_labels.csv')
dir_ = '/faster_rcnn_data/grayscale_train'
pictures_files = df['filename'].unique()
all_gt_boxes = []
images = []
for picture in pictures_files:
    current_df = df[df['filename']==picture]
    gt_boxes = []
    for i in range(current_df.shape[0]):
        gt_boxes.append(current_df.iloc[i,4:].values)
    all_gt_boxes.append(gt_boxes)
    
    img = cv2.imread(dir_ + "/" + picture,cv2.IMREAD_GRAYSCALE)
    images.append(np.expand_dims(np.array(img,dtype=np.float32),axis=2))

data = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(images,all_gt_boxes),
                                      output_types=(tf.float32, tf.float32),
                                      output_shapes=(tf.TensorShape([None, None, 1]), 
                                                     tf.TensorShape([None,None])))
data = data.repeat(5)
iterator = data.make_one_shot_iterator()
next_element = iterator.get_next()

saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
assign_op, feed_dict_init = initialize_feature_extractor_weights(
    '/faster_rcnn_data/ssd_mobilenet_v1.npy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign_op, feed_dict_init)
    for epoch in range(5):
        for i in range(115):
            print(i)
            feed_img,feed_gt = sess.run(next_element)
            feed_img = np.expand_dims(feed_img,axis=0)
            loss1 = sess.run(total_loss, feed_dict={img_to_feed: feed_img, gt_boxes_to_feed: feed_gt})
            print('total_loss,rcnn_bbox_loss:',loss1)
            _, = sess.run([train_op],feed_dict={img_to_feed: feed_img, gt_boxes_to_feed: feed_gt})
        saver.save(sess,'./my_model.ckpt',global_step=epoch)

#         print('rpn_bboxes_target',rpn1.shape)
#         print('rpn_labels',rpn2.shape)
#         print('rpn_indices_true_bboxes',rpn3.shape)
#         ########

#         rcnn1,rcnn2,rcnn3 = sess.run([rcnn_bbox_target, rcnn_labels, rcnn_indices_true_bboxes], feed_dict={img_to_feed: feed_img,
#                                                                                             gt_boxes_to_feed: feed_gt})

#         print('rcnn_bbox_target', rcnn1.shape)
#         print('rcnn_labels', rcnn2.shape)
#         print('rcnn_indices_true_bboxes',rcnn3[:4])

#         loss1,loss2,loss3,loss4,cond_ = sess.run([rcnn_bbox_loss,rcnn_cross_entropy, rpn_cross_entropy, rpn_bbox_loss,cond], feed_dict={img_to_feed: feed_img,
#                                                                                                     gt_boxes_to_feed: feed_gt})
#         print('!!!!')
#         print('rpn_cross_entropy', loss3)
#         print('rpn_bbox_loss', loss4)
#         print('rcnn_cross_entropy',loss2)
#         print('rcnn_bbox_loss',loss1)
#         print('cond',cond_)

#         print('__________________________')