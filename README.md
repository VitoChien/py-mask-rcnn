# py-mask-rcnn

Mask R-CNN in Caffe

This repo attempts to reproduce Mask R-CNN in Caffe.

But due to the memory managerment of Caffe, the number of proposal output by rpn should be set to very small, which may impact the performance of the model.
