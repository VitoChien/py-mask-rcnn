from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections
import caffe
from caffe import layers as L, params as P, to_proto
from .layers import *

class ResNet(): 
    def __init__(self, stages=[3, 4, 6, 3], channals=64, deploy=False, classes = 2, feat_stride = 16, \
                 pooled_size=[14, 14], out_size=[28, 28], module = "normal", pooling = "align", scales=[4, 8, 16, 32], ratio=[0.5, 1, 2], rois_num=128):
        self.stages = stages
        self.channals = channals
        self.deploy = deploy
        self.classes = classes
        self.anchors = len(scales) * len(ratio)
        self.feat_stride = feat_stride
        self.module = module
        self.net = caffe.NetSpec()
        self.pooling = pooling
        self.pooled_w = pooled_size[0] 
        self.pooled_h = pooled_size[1]
        self.out_w = out_size[0]
        self.out_h = out_size[1]
        self.scales =scales
        self.ratio =ratio
        self.rois_num = rois_num


    # def resnet_rcnn(self):
    #     channals = self.channals
    #     if not self.deploy:
    #         data, im_info, gt_boxes = data_layer_train(self.net, self.classes, self.out_h)
    #     else:
    #         data, im_info = data_layer_test()
    #         gt_boxes = None
    #     conv1 = conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True)
    #     pool1 = pooling_layer(3, 2, 'MAX', 'pool1', conv1)
    #     k=0
    #     index = 1
    #     out = pool1
    #     if self.module == "normal":
    #         residual_block = self.residual_block
    #     else:
    #         residual_block = self.residual_block_basic
    #
    #     for i in self.stages[:-1]:
    #         index += 1
    #         for j in range(i):
    #             if j == 0:
    #                 if index == 2:
    #                     stride = 1
    #                 else:
    #                     stride = 2
    #                 out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
    #             else:
    #                 out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
    #         channals *= 2
    #
    #
    #     if not self.deploy:
    #         rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data, fixed=True)
    #         rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
    #             self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)
    #     else:
    #         rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
    #         rois, scores = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)
    #
    #
    #     feat_aligned = self.roi_align(out, rois)
    #     out = feat_aligned
    #
    #     index += 1
    #     for j in range(self.stages[-1]):
    #         if j == 0:
    #             stride = 1
    #             out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
    #         else:
    #             out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals)
    #     pool5 = self.ave_pool(7, 1, "pool5", out)
    #     cls_score, bbox_pred = self.final_cls_bbox(pool5)
    #
    #     if not self.deploy:
    #         self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight= 1, propagate_down=[1,0])
    #         self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,\
    #                             loss_weight= 1)
    #     else:
    #         self.net["cls_prob"] =  L.Softmax(cls_score)
    #     return self.net.to_proto()

    def resnet_mask_end2end(self):
        channals = self.channals
        if not self.deploy:
            data, im_info, gt_boxes, ins = \
                data_layer_train_with_ins(self.net, self.classes, with_rpn=True)
        else:
            data, im_info = data_layer_test(self.net)
            gt_boxes = None
        conv1 = conv_factory(self.net, "conv1", data, 7, channals, 2, 3, bias_term=True)
        pool1 = pooling_layer(self.net, 3, 2, 'MAX', 'pool1', conv1)
        index = 1
        out = pool1
        if self.module == "normal":
            residual_block = residual_block
        else:
            residual_block = residual_block_basic

        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    out = residual_block(self.net, "res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    out = residual_block(self.net, "res" + str(index) + ascii_lowercase[j], out, channals)
            channals *= 2
        if not self.deploy:
            rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = rpn(self.net, out, gt_boxes, im_info, data, fixed=False)
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mask_roi, masks = \
                roi_proposals(self.net, rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)
            self.net["rois_cat"] = L.Concat(rois,mask_roi, name="rois_cat", axis=0)
            rois=self.net["rois_cat"]
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = rpn(self.net, out, gt_boxes, im_info, data)
            rois, scores = roi_proposals(self.net, rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        feat_out = out

        feat_aligned = roi_align(self.net, "det_mask", feat_out, rois)
        # if not self.deploy:
        #     self.net["silence_mask_rois"] = L.Silence(mask_rois, ntop=0)
        # if not self.deploy:
        #     mask_feat_aligned = self.roi_align("mask", feat_out, mask_rois)
        # else:
        #     mask_feat_aligned = self.roi_align("mask", feat_out, rois)
        out = feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j == 0:
                stride = 1
                out = residual_block(self.net, "res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                out = residual_block(self.net, "res" + str(index) + ascii_lowercase[j], out, channals)

        if not self.deploy:
            self.net["det_feat"], self.net["mask_feat"] = L.Slice(self.net, out, ntop=2, name='slice', slice_param=dict(slice_dim=0, slice_point=self.rois_num))
            feat_mask = self.net["mask_feat"]
            out = self.net["det_feat"]

        # for bbox detection
        pool5 = ave_pool(self.net, 7, 1, "pool5",  out)
        cls_score, bbox_pred = final_cls_bbox(self.net, pool5)

        if not self.deploy:
            self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight=1, propagate_down=[1, 0])
            self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
                                                   loss_weight=1)
        else:
            self.net["cls_prob"] = L.Softmax(cls_score)


        # # for mask prediction
        if not self.deploy:
            mask_feat_aligned = feat_mask
        else:
            mask_feat_aligned = out
        # out = mask_feat_aligned
        out = L.Deconvolution(mask_feat_aligned, name = "mask_deconv1",convolution_param=dict(kernel_size=2, stride=2,
                                            num_output=256, pad=0, bias_term=False,
                                            weight_filler=dict(type='msra')))
        out = L.BatchNorm(out, name="bn_mask_deconv1",in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
        out = L.Scale(out, name = "scale_mask_deconv1", in_place=True, scale_param=dict(bias_term=True))
        out = L.ReLU(out, name="mask_deconv1_relu", in_place=True)
        mask_out = conv_factory(self.net, "mask_out", out, 1, self.classes-1, 1, 0, bias_term=True)
        # for i in range(4):
        #     out = self.conv_factory("mask_conv"+str(i), out, 3, 256, 1, 1, bias_term=False)
        # mask_out = self.conv_factory("mask_out", out, 1, 1, 1, 0, bias_term=False)

        if not self.deploy:
            self.net["loss_mask"] = L.SigmoidCrossEntropyLoss(mask_out, masks, loss_weight=1, propagate_down=[1, 0],
                                                      loss_param=dict(
                                                          normalization=1,
                                                          ignore_label = -1
                                                      ))
        else:
            self.net["mask_prob"] = L.Sigmoid(mask_out)

        return self.net.to_proto()

    def resnet_mask_rcnn_rpn(self, stage=1):
        channals = self.channals
        if not self.deploy:
            data, im_info, gt_boxes = self.data_layer_train()
        else:
            data, im_info = self.data_layer_test()
            gt_boxes = None
        if stage == 1:
            pre_traned_fixed = True
        else:
            pre_traned_fixed = False
        conv1 = self.conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True, fixed=pre_traned_fixed)
        pool1 = self.pooling_layer(3, 2, 'MAX', 'pool1', conv1)
        index = 1
        out = pool1
        if self.module == "normal":
            residual_block = self.residual_block
        else:
            residual_block = self.residual_block_basic

        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
                else:
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
            channals *= 2

        if not self.deploy:
            rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois, scores = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        if not self.deploy:
            self.net["dummy_roi_pool_conv5"] = L.DummyData(name = "dummy_roi_pool_conv5", shape=[dict(dim=[1,channals*2,14,14])])
            out = self.net["dummy_roi_pool_conv5"]
            index += 1
            for j in range(self.stages[-1]):
                if j == 0:
                    stride = 1
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals)
            if stage==1:
                self.net["silence_res"] = L.Silence(out, ntop=0)

            if stage==2:
                # for bbox detection
                pool5 = self.ave_pool(7, 1, "pool5", out)
                cls_score, bbox_pred = self.final_cls_bbox(pool5)
                self.net["silence_cls_score"] = L.Silence(cls_score, ntop=0)
                self.net["silence_bbox_pred"] = L.Silence(bbox_pred, ntop=0)

                # for mask prediction
                mask_conv1 = self.conv_factory("mask_conv1", out, 3, 256, 1, 1, bias_term=True)
                mask_out = self.conv_factory("mask_out", mask_conv1, 1, self.classes, 1, 0, bias_term=True)
                self.net["silence_mask_out"] = L.Silence(mask_out, ntop=0)
        return self.net.to_proto()

    def resnet_mask_rcnn_mask_rcnn(self, stage=1):
        channals = self.channals
        if not self.deploy:
            data, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mask_rois, masks = \
                self.data_layer_train_with_ins(with_rpn=False)
            im_info = None
        else:
            data, im_info = self.data_layer_test()
        gt_boxes = None
        if stage == 1:
            pre_traned_fixed = False
        else:
            pre_traned_fixed = True
        conv1 = self.conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True, fixed=pre_traned_fixed)
        pool1 = self.pooling_layer(3, 2, 'MAX', 'pool1', conv1)
        index = 1
        out = pool1
        if self.module == "normal":
            residual_block = self.residual_block
        else:
            residual_block = self.residual_block_basic

        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
                else:
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
            channals *= 2

        if not self.deploy:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data, fixed=True)
            self.net["silence_rpn_cls_score_reshape"] = L.Silence(rpn_cls_score_reshape, ntop=0)
            self.net["silence_rpn_bbox_pred"] = L.Silence(rpn_bbox_pred, ntop=0)
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois, scores = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        feat_out = out

        if not self.deploy:
            self.net["rois_cat"] = L.Concat(rois, mask_rois, name="rois_cat", axis=0)
            rois=self.net["rois_cat"]

        feat_aligned = self.roi_align("det_mask", feat_out, rois)
        # if not self.deploy:
        #     self.net["silence_mask_rois"] = L.Silence(mask_rois, ntop=0)
        # if not self.deploy:
        #     mask_feat_aligned = self.roi_align("mask", feat_out, mask_rois)
        # else:
        #     mask_feat_aligned = self.roi_align("mask", feat_out, rois)
        out = feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j == 0:
                stride = 1
                out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals)

        if not self.deploy:
            self.net["det_feat"], self.net["mask_feat"] = L.Slice(out, ntop=2, name='slice', slice_param=dict(slice_dim=0, slice_point=self.rois_num))
            feat_mask = self.net["mask_feat"]
            out = self.net["det_feat"]

        # for bbox detection
        pool5 = self.ave_pool(7, 1, "pool5",  out)
        cls_score, bbox_pred = self.final_cls_bbox(pool5)

        if not self.deploy:
            self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight=1, propagate_down=[1, 0])
            self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
                                                   loss_weight=1)
        else:
            self.net["cls_prob"] = L.Softmax(cls_score)


        # # for mask prediction
        if not self.deploy:
            mask_feat_aligned = feat_mask
        else:
            mask_feat_aligned = out
        # out = mask_feat_aligned
        out = L.Deconvolution(mask_feat_aligned, name = "mask_deconv1",convolution_param=dict(kernel_size=2, stride=2,
                                            num_output=256, pad=0, bias_term=False,
                                            weight_filler=dict(type='msra'),
                                            bias_filler=dict(type='constant')))
        out = L.BatchNorm(out, name="bn_mask_deconv1",in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
        out = L.Scale(out, name = "scale_mask_deconv1", in_place=True, scale_param=dict(bias_term=True))
        out = L.ReLU(out, name="mask_deconv1_relu", in_place=True)
        mask_out = self.conv_factory("mask_out", out, 1, self.classes-1, 1, 0, bias_term=True)
        # for i in range(4):
        #     out = self.conv_factory("mask_conv"+str(i), out, 3, 256, 1, 1, bias_term=False)
        # mask_out = self.conv_factory("mask_out", out, 1, 1, 1, 0, bias_term=False)

        if not self.deploy:
            self.net["loss_mask"] = L.SigmoidCrossEntropyLoss(mask_out, masks, loss_weight=1, propagate_down=[1, 0],
                                                      loss_param=dict(
                                                          normalization=1,
                                                          ignore_label = -1
                                                      ))
        else:
            self.net["mask_prob"] = L.Sigmoid(mask_out)

        return self.net.to_proto()

    def resnet_mask_rcnn_test(self):
        channals = self.channals
        data, rois = self.data_layer_test(with_roi=True)
        pre_traned_fixed = True
        conv1 = self.conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True, fixed=pre_traned_fixed)
        pool1 = self.pooling_layer(3, 2, 'MAX', 'pool1', conv1)
        index = 1
        out = pool1
        if self.module == "normal":
            residual_block = self.residual_block
        else:
            residual_block = self.residual_block_basic

        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
                else:
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
            channals *= 2

        mask_feat_aligned = self.roi_align("mask", out, rois)
        out = mask_feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j == 0:
                stride = 1
                out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals)

        # for mask prediction
        out = L.Deconvolution(out, name = "mask_deconv1",convolution_param=dict(kernel_size=2, stride=2,
                                    num_output=256, pad=0, bias_term=False,
                                    weight_filler=dict(type='msra'),
                                    bias_filler=dict(type='constant')))
        out = L.BatchNorm(out, name="bn_mask_deconv1",in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
        out = L.Scale(out, name = "scale_mask_deconv1", in_place=True, scale_param=dict(bias_term=True))
        out = L.ReLU(out, name="mask_deconv1_relu", in_place=True)
        mask_out = self.conv_factory("mask_out", out, 1, self.classes-1, 1, 0, bias_term=True)
        self.net["mask_prob"] = L.Sigmoid(mask_out)

        return self.net.to_proto()

def main():
    rois_num = 16
    scales = [2, 4, 8, 16, 32]
    # resnet_rpn_test = ResNet(deploy=True, scales = scales)
    # resnet_rpn_train_1 = ResNet(deploy=False, scales = scales)
    # resnet_rpn_train_2 = ResNet(deploy=False, scales = scales)
    # resnet_mask_test = ResNet(deploy=True, scales = scales)
    # resnet_mask_train_1 = ResNet(deploy=False, scales = scales, rois_num=rois_num)
    # resnet_mask_train_2 = ResNet(deploy=False, scales = scales, rois_num=rois_num)
    # resnet_mask_test_mask = ResNet(deploy=True, scales = scales)
    resnet_mask_end2end_train = ResNet(deploy=False, scales = scales, rois_num=rois_num)
    resnet_mask_end2end_test = ResNet(deploy=True, scales = scales)
    #for net in ('18', '34', '50', '101', '152'):
    # with open('stage1_rpn_train.pt', 'w') as f:
    #     f.write(str(resnet_rpn_train_1.resnet_mask_rcnn_rpn(stage=1)))
    # with open('stage2_rpn_train.pt', 'w') as f:
    #     f.write(str(resnet_rpn_train_2.resnet_mask_rcnn_rpn(stage=2)))
    # with open('stage1_mask_rcnn_train.pt', 'w') as f:
    #     f.write(str(resnet_mask_train_1.resnet_mask_rcnn_mask_rcnn(stage=1)))
    # with open('stage2_mask_rcnn_train.pt', 'w') as f:
    #     f.write(str(resnet_mask_train_2.resnet_mask_rcnn_mask_rcnn(stage=2)))
    # with open('mask_rcnn_test.pt', 'w') as f:
    #     f.write(str(resnet_mask_test.resnet_mask_rcnn_mask_rcnn()))
    # with open('mask_rcnn_mask_test.pt', 'w') as f:
    #     f.write(str(resnet_mask_test_mask.resnet_mask_rcnn_test()))
    # with open('rpn_test.pt', 'w') as f:
    #     f.write(str(resnet_rpn_test.resnet_mask_rcnn_rpn()))
    with open('resnet_mask_end2end.pt', 'w') as f:
        f.write(str(resnet_mask_end2end_train.resnet_mask_end2end()))
    with open('resnet_mask_end2end_test.pt', 'w') as f:
        f.write(str(resnet_mask_end2end_test.resnet_mask_end2end()))

if __name__ == '__main__':
    main()
