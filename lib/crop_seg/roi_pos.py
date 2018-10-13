import caffe
import numpy as np


class RoiPosLayer(caffe.Layer):
    """
    Keep the rois that containing positive exmaples
    """

    def setup(self, bottom, top):
        pool5 = bottom[0].data
        top[0].reshape(*pool5.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pool5 = bottom[0].data
        labels = bottom[1].data
        # print 'pool5.shape', pool5.shape
        # print 'labels.shape', labels.shape
        # print 'labels', labels
        pos_ind = np.where(labels > 0)
        pool5 = pool5[pos_ind]
        # print 'pool5.shape', pool5.shape
        top[0].reshape(*pool5.shape)
        top[0].data[...] = pool5

    def backward(self, top, propagate_down, bottom):
        pass
