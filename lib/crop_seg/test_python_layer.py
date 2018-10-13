import caffe
import numpy as np
import cv2


class TestPythonLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        # if len(bottom) != 1:
        #     raise Exception("Need two inputs to compute distance.")
        # seg = seg[:, :1, :, :]

        seg = bottom[0].data
        print "setup:"
        print seg.shape

    def reshape(self, bottom, top):
        # loss output is scalar
        seg = bottom[0].data
        top[0].reshape(*(seg.shape))

    def forward(self, bottom, top):
        # top[0].data[...] = np.sum(bottom[0].data**2) / bottom[0].num / 2.
        seg = bottom[0].data
        # print "ori :"
        # print seg
        seg = np.uint8(seg)
        # print "after :"
        # print seg
        top[0].data[...] = seg

        # print "7777777777777777"
        # print top[0].data[:1, :1, 307, 322]
        # cv2.imwrite("/home/chen/my.png", top[0].data)
        # print 'endendend'

    def backward(self, top, propagate_down, bottom):
        pass