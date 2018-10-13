import caffe
import numpy as np


class GtRoisLayer(caffe.Layer):
    """
    Keep the rois that containing positive exmaples
    """

    def setup(self, bottom, top):
        gt_boxes = bottom[0].data
        top[0].reshape(*gt_boxes.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        gt_boxes = bottom[0].data
        gt_boxes = gt_boxes[:, :4]
        batch_inds = np.zeros(gt_boxes.shape[0])
        gt_boxes = np.c_[batch_inds, gt_boxes]
        top[0].reshape(*gt_boxes.shape)
        top[0].data[...] = gt_boxes

    def backward(self, top, propagate_down, bottom):
        pass
