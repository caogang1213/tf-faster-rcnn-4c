from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/opt/tf-faster-rcnn/tools')
sys.path.insert(0, '/opt/opencv/lib/python2.7/site-packages/')

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from operator import mul
from utils.timer import Timer

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
import glob
import re
import errno

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from model.config import cfg

CLASSES = ('__background__','calyx', 'stem', 'defect', 'other')
COLORS = ('red', 'blue', 'yellow', 'green')

def bbox_visualization(img, scores, boxes):

    NMS_THRESH = 0.3

    # check 3 channel or 4 channel
    # convert BRG to RGB
    if cfg.IMGCHANNEL == 3:
        img = img[:,:,(2,1,0)]
    elif cfg.IMGCHANNEL == 4:
        img = img[:,:,(2, 1, 0, 3)]
    else:
        raise NotImplementedError

    # image canvas
    fig, ax = plt.subplots()
    ax.imshow(img, aspect='equal')

    for cls_idx, cls in enumerate(CLASSES[1:]):
        cls_idx += 1 # start from 0, and skip 'background' class
        cls_boxes = boxes[:, 4*cls_idx:4*(cls_idx + 1)]  # obtain bounding box (x, y, w, h)
        cls_scores = scores[:, cls_idx]

        tupleScoreBox = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        selectedIndex = nms(tupleScoreBox, NMS_THRESH)
        threshedScoreBox = tupleScoreBox[selectedIndex, :]

        thresh = 0.5

        if cls == 'calyx' or cls == 'stem':
            idex = np.where(max(threshedScoreBox[:, -1]) >= thresh)[0]
        else:
            idex = np.where(threshedScoreBox[:, -1] >= thresh)[0]

        if len(idex) == 0:
            0
        else:
            for i in idex:
                bbox = threshedScoreBox[
                    i, :4]
                score = threshedScoreBox[i, -1]

                # plot bounding boxes and label scores
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                    fill=False,
                    edgecolor=COLORS[cls_idx - 1],
                    linewidth=1)
                    )
                ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor=COLORS[cls_idx - 1], alpha=0.8),
                fontsize=10, color='black')

def tf_faster_rcnn_predict(sess, net, imagename):
    if cfg.IMGCHANNEL == 3:
        img = cv2.imread(imagename)
    elif cfg.IMGCHANNEL == 4:
        img = cv2.imread(imagename, cv2.IMREAD_UNCHANGED)
    else:
        raise NotImplementedError

    # detect objects and regress bounding boxes
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, img)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    bbox_visualization(img, scores, boxes)

def tf_params_counting():
    count = 0
    total_parameters = 0
    print('============================================================')
    print('The number of parameters to be learnt for Faster-RCNN Model.')
    # for variable in tf.trainable_variables():
    for variable in tf.global_variables():
        # shape is an array of tf.Dimension
        count += 1
        shape = variable.get_shape()
        name = variable.name
        variable_parameters = reduce(mul, shape)
        print('--------')
        print('TF Layer %s: name=%s, shape=%s, num of params=%s' 
               % (count, name, shape, variable_parameters))
        total_parameters += variable_parameters.value
    print('--------')
    print('Total:%s'% total_parameters)
    print('============================================================')

if __name__ == '__main__':

    codedescription = 'Faster-RCNN Model for Blueberry Calyx Recognition and Detection'

    parser = argparse.ArgumentParser(description=codedescription)
    parser.add_argument("--folderdir", required=True, help="Path to the image folder")
    parser.add_argument("--tfnet", required=True, help="Deep NN model: res101 or vgg16") 
    parser.add_argument("--tfmodel", required=True, help="Path to the Faster-RCNN model file")
    parser.add_argument("--savedir", required=True, help="Path to save results")
    args = parser.parse_args()

    tfmodel = args.tfmodel

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\n Have you trained your Faster-RCNN model '
        'with your own dataset?').format(tfmodel + '.meta'))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    sess = tf.Session(config=tfconfig)

    if args.tfnet == 'vgg16':
        net = vgg16()
    elif args.tfnet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError

    # 5 denotes the number of classes during the training of Faster R-CNN
    # CLASSES can remove one of classes because we do not want detect and show it
    # But the number '5' cannot be changed after training
    # otherwise the following error happens
    # --> Assign requires shapes of both tensors to match. 
    # lhs shape= [...] rhs shape= [...]
    net.create_architecture("TEST",5,
            tag='default', anchor_scales=[4, 8, 16, 32])

    tfsaver = tf.train.Saver()
    tfsaver.restore(sess, tfmodel)

    tf_params_counting()

    print('=========================================')
    print('Successfully loaded pretrained Faster-RCNN {:s}'.format(tfmodel))

    BatchImageNames = os.listdir(args.folderdir)
    BatchImageNames = sorted(BatchImageNames, key=lambda x:(int(re.sub('\D','',x)),x))
    savedir = os.path.join(args.savedir)
    try:
        os.makedirs(savedir)
    except OSError as e:
        if e.errno != errno.EEXIST:
                raise
    count = 0
    print('----------')
    for tmpImg in BatchImageNames[:1000]:
        imgdir = os.path.join(args.folderdir, tmpImg)
        tf_faster_rcnn_predict(sess, net, imgdir)
        count += 1
        # plt.show()
        saveImgdir = os.path.join(savedir, tmpImg)
        plt.savefig(saveImgdir)
        plt.close()
    print('----------')
    print('%s prediction results have been saved '%(count))
    print('========================================')
