from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections
import caffe
from caffe import layers as L, params as P, to_proto



class ResNet(): 
    def __init__(self, stages=[3, 4, 6, 3], channals=64, deploy=False, classes = 2, anchors = 9, feat_stride = 16, pooled_size=[14, 14], module = "normal", pooling = "align"):
        self.stages = stages
        self.channals = channals
        self.deploy = deploy
        self.classes = classes
        self.anchors = anchors
        self.feat_stride = feat_stride
        self.module = module
        self.net = caffe.NetSpec()
        self.pooling = pooling
        self.pooled_w = pooled_size[0] 
        self.pooled_h = pooled_size[1] 

    def roi_align(self, name, bottom, roi):
        if self.pooling == "align":
            self.net["ROIAlign" + name] = L.ROIAlign(bottom, roi, roi_align_param = {
                    "pooled_w": self.pooled_w,
                    "pooled_h": self.pooled_h,
                    "spatial_scale": 1/float(self.feat_stride)})
            return self.net["ROIAlign" + name]
        else:
            self.net["ROIPooling"] = L.ROIPooling(bottom, roi, roi_pooling_param = {
                    "pooled_w": self.pooled_w,
                    "pooled_h": self.pooled_h,
                    "spatial_scale": 1/float(self.feat_stride)})
            return self.net["ROIPooling"]

    def conv_factory(self, name, bottom, ks, nout, stride=1, pad=0, bias_term=False, fixed = False):
        if not fixed:
            self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'))
        else:
            if bias_term:
                self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'),
                                               param = [{'lr_mult':0, 'decay_mult':0},{'lr_mult':0, 'decay_mult':0}])
            else:
                self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'),
                                               param = [{'lr_mult':0, 'decay_mult':0}])
        if "res" in name:
            self.net[name.replace("res", "bn")] = L.BatchNorm(self.net[name], in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
            self.net[name.replace("res", "scale")]  = L.Scale(self.net[name.replace("res", "bn")], in_place=True, scale_param=dict(bias_term=True))
            self.net[name + "_relu"] = L.ReLU(self.net[name.replace("res", "scale")], in_place=True)
        else:
            self.net["bn_" + name] = L.BatchNorm(self.net[name], in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
            self.net["scale_" + name] = L.Scale(self.net["bn_" + name], in_place=True, scale_param=dict(bias_term=True))
            self.net[name + "_relu"] = L.ReLU(self.net["scale_" + name], in_place=True)
        return self.net[name + "_relu"]

    def conv_factory_inverse_no_relu(self, name, bottom, ks, nout, stride=1, pad=0, deploy=False, bias_term=False, fixed = False):
        if not fixed:
            self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'))
        else:
            if bias_term:
                self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'),
                                               param = [{'lr_mult':0, 'decay_mult':0},{'lr_mult':0, 'decay_mult':0}])
            else:
                self.net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                        num_output=nout, pad=pad, bias_term=bias_term, weight_filler=dict(type='msra'),
                                               param = [{'lr_mult':0, 'decay_mult':0}])
        if "res" in name:
            self.net[name.replace("res", "bn")] = L.BatchNorm(self.net[name], in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
            self.net[name.replace("res", "scale")]  = L.Scale(self.net[name.replace("res", "bn")], in_place=True, scale_param=dict(bias_term=True))
            return self.net[name.replace("res", "scale")]
        else:
            self.net["bn_" + name] = L.BatchNorm(self.net[name], in_place=True, batch_norm_param=dict(use_global_stats=self.deploy))
            self.net["scale_" + name] = L.Scale(self.net["bn_" + name], in_place=True, scale_param=dict(bias_term=True))
            return self.net["scale_" + name]

    def rpn(self, bottom, gt_boxes, im_info, data, fixed = False):
        if not fixed:
            self.net["rpn_conv/3x3"] = L.Convolution(bottom, kernel_size=3, stride=1,
                                        num_output=512, pad=1,
                                        param= [{'lr_mult':1},{'lr_mult':2}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
        else:
            self.net["rpn_conv/3x3"] = L.Convolution(bottom, kernel_size=3, stride=1,
                                        num_output=512, pad=1,
                                        param= [{'lr_mult':0},{'lr_mult':0}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
        self.net["rpn_relu/3x3"] = L.ReLU(self.net["rpn_conv/3x3"] , in_place=True)
        if not fixed:
            self.net["rpn_cls_score"] = L.Convolution(self.net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                        num_output= 2 * self.anchors, pad=0,
                                        param= [{'lr_mult':1},{'lr_mult':2}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
            self.net["rpn_bbox_pred"] = L.Convolution(self.net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                        num_output= 4 * self.anchors, pad=0,
                                        param= [{'lr_mult':1},{'lr_mult':2}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
        else:
            self.net["rpn_cls_score"] = L.Convolution(self.net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                        num_output= 2 * self.anchors, pad=0,
                                        param= [{'lr_mult':0},{'lr_mult':0}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
            self.net["rpn_bbox_pred"] = L.Convolution(self.net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                        num_output= 4 * self.anchors, pad=0,
                                        param= [{'lr_mult':0},{'lr_mult':0}],
                                        weight_filler=dict(type='gaussian', std=0.01),
                                        bias_filler=dict(type='constant', value=0))
        self.net["rpn_cls_score_reshape"] = L.Reshape(self.net["rpn_cls_score"],
                                    reshape_param= {"shape" : { "dim": [0, 2, -1, 0]}})

        if (not self.deploy) and (not fixed):
            self.net["rpn_labels"], self.net["rpn_bbox_targets"], self.net["rpn_bbox_inside_weights"], self.net["rpn_bbox_outside_weights"] = \
                        L.Python(self.net["rpn_cls_score"], gt_boxes, im_info, data,
                            name = 'rpn-data',
                            python_param=dict(
                                            module='rpn.anchor_target_layer',
                                            layer='AnchorTargetLayer',
                                            param_str='"feat_stride": %s' %(self.feat_stride)),
                            ntop=4,)
            self.net["rpn_cls_loss"] = L.SoftmaxWithLoss(self.net["rpn_cls_score_reshape"], self.net["rpn_labels"], name = "rpn_loss_cls", propagate_down=[1,0],\
                            loss_weight = 1, loss_param = {"ignore_label": -1, "normalize": True})
            self.net["rpn_loss_bbox"] = L.SmoothL1Loss(self.net["rpn_bbox_pred"], self.net["rpn_bbox_targets"], \
                            self.net["rpn_bbox_inside_weights"], self.net["rpn_bbox_outside_weights"], \
                            name= "loss_bbox", loss_weight = 1, smooth_l1_loss_param = {"sigma": 3.0})
            return self.net["rpn_cls_loss"], self.net["rpn_loss_bbox"], self.net["rpn_cls_score_reshape"], self.net["rpn_bbox_pred"]
        else:
            return self.net["rpn_cls_score_reshape"], self.net["rpn_bbox_pred"]
        

    def roi_proposals(self, rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes):
        self.net["rpn_cls_prob"] = L.Softmax(rpn_cls_score_reshape, name = "rpn_cls_prob")
        self.net["rpn_cls_prob_reshape"] = L.Reshape(self.net["rpn_cls_prob"], name = "rpn_cls_prob_reshape", \
                    reshape_param= {"shape" : { "dim": [0, 18, -1, 0]}})
        
        if not self.deploy:
            self.net["rpn_rois"] = L.Python(self.net["rpn_cls_prob_reshape"], rpn_bbox_pred, im_info, 
                            name = 'proposal',
                            python_param=dict(
                                            module='rpn.proposal_layer',
                                            layer='ProposalLayer',
                                            param_str='"feat_stride": %s' %(self.feat_stride)),
                            ntop=1,)
            self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], self.net["bbox_outside_weights"] = \
                        L.Python(self.net["rpn_rois"], self.net["gt_boxes"],
                            name = 'roi-data',
                            python_param=dict(
                                            module='rpn.proposal_target_layer',
                                            layer='ProposalTargetLayer',
                                            param_str='"num_classes": %s' %(self.classes)),
                            ntop=5,)
            return self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], self.net["bbox_outside_weights"]
        else:
            self.net["rois"],self.net["scores"] = L.Python(self.net["rpn_cls_prob_reshape"], rpn_bbox_pred, im_info,
                            name = 'proposal',
                            python_param=dict(
                                            module='rpn.proposal_layer',
                                            layer='ProposalLayer',
                                            param_str='"feat_stride": %s' %(self.feat_stride)),
                            ntop=2,)
            return self.net["rois"], self.net["scores"]


    def final_cls_bbox(self, bottom, fixed = False):
        if not fixed:
            self.net["cls_score"] = L.InnerProduct(bottom, name = "cls_score",
                                    num_output= self.classes,
                                    param= [{'lr_mult':1},{'lr_mult':2}],
                                    weight_filler=dict(type='gaussian', std=0.001),
                                    bias_filler=dict(type='constant', value=0))
            self.net["bbox_pred"] = L.InnerProduct(bottom, name = "bbox_pred",
                                    num_output= 4 * self.classes,
                                    param= [{'lr_mult':1},{'lr_mult':2}],
                                    weight_filler=dict(type='gaussian', std=0.001),
                                    bias_filler=dict(type='constant', value=0))
        else:
            self.net["cls_score"] = L.InnerProduct(bottom, name = "cls_score",
                                    num_output= self.classes,
                                    param= [{'lr_mult':0},{'lr_mult':0}],
                                    weight_filler=dict(type='gaussian', std=0.001),
                                    bias_filler=dict(type='constant', value=0))
            self.net["bbox_pred"] = L.InnerProduct(bottom, name = "bbox_pred",
                                    num_output= 4 * self.classes,
                                    param= [{'lr_mult':0},{'lr_mult':0}],
                                    weight_filler=dict(type='gaussian', std=0.001),
                                    bias_filler=dict(type='constant', value=0))
        return self.net["cls_score"], self.net["bbox_pred"]

    def data_layer_train(self,with_rpn=True):
        if not self.deploy:
            if with_rpn:
                self.net["data"], self.net["im_info"], self.net["gt_boxes"]= L.Python(
                                    name = 'input-data',
                                    python_param=dict(
                                                    module='roi_data_layer.layer',
                                                    layer='RoIDataLayer',
                                                    param_str='"num_classes": %s' %(self.classes)),
                                    ntop=3,)
                return self.net["data"], self.net["im_info"], self.net["gt_boxes"]
            else:
                self.net["data"], self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], \
                self.net["bbox_outside_weights"]= L.Python(
                                    name = 'input-data',
                                    python_param=dict(
                                                    module='roi_data_layer.layer',
                                                    layer='RoIDataLayer',
                                                    param_str='"num_classes": %s' %(self.classes)),
                                    ntop=3,)
                return self.net["data"], self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], \
                        self.net["bbox_outside_weights"]


    def data_layer_test(self):
        if self.deploy:
            self.net["data"] = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
            self.net["im_info"] = L.Input(shape=[dict(dim=[1, 3])])
        return self.net["data"], self.net["im_info"]

    def data_layer_train_with_ins(self, with_rpn=False):
        if not self.deploy:
            if with_rpn:
                self.net["data"], self.net["im_info"], self.net["gt_boxes"], self.net["mask_rois"], self.net["masks"]= L.Python(
                                    name = 'input-data',
                                    python_param=dict(
                                                    module='roi_data_layer.layer',
                                                    layer='RoIDataLayer',
                                                    param_str='"num_classes": %s' %(self.classes)),
                                    ntop=5,)
                return self.net["data"], self.net["im_info"], self.net["gt_boxes"], self.net["mask_rois"], self.net["masks"]
            else:
                self.net["data"], self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], \
                self.net["bbox_outside_weights"], self.net["mask_rois"], self.net["masks"] = L.Python(
                                    name = 'input-data',
                                    python_param=dict(
                                                    module='roi_data_layer.layer',
                                                    layer='RoIDataLayer',
                                                    param_str='"num_classes": %s' %(self.classes)),
                                    ntop=8,)
                return self.net["data"], self.net["rois"], self.net["labels"], self.net["bbox_targets"], self.net["bbox_inside_weights"], \
                        self.net["bbox_outside_weights"], self.net["mask_rois"], self.net["masks"]

    def pooling_layer(self, kernel_size, stride, pool_type, layer_name, bottom):
        self.net[layer_name] = L.Pooling(bottom, pool=eval("P.Pooling." + pool_type), kernel_size=kernel_size, stride=stride)
        return self.net[layer_name]

    def ave_pool(self, kernel_size, stride, layer_name, bottom):
        return self.pooling_layer(kernel_size, stride, 'AVE', layer_name, bottom)

    def residual_block_shortcut(self, name, bottom, num_filter, stride=1, fixed = False):
        conv1 = self.conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, fixed = fixed)
        conv2 = self.conv_factory(name + "_branch2b", conv1, 3, num_filter, stride, 1, fixed = fixed)
        conv3 = self.conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, stride, 0, fixed = fixed)
        self.net[name] = L.Eltwise(bottom, conv3, operation=P.Eltwise.SUM)
        self.net[name + "_relu"] = L.ReLU(self.net[name], in_place=True)
        return self.net[name + "_relu"]

    def residual_block(self, name, bottom, num_filter, stride=1, fixed = False):
        conv1 = self.conv_factory(name + "_branch2a",bottom, 1, num_filter, stride, 0, fixed = fixed)
        conv2 = self.conv_factory(name + "_branch2b",conv1, 3, num_filter, 1, 1, fixed = fixed)
        conv3 = self.conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, 1, 0, fixed = fixed)
        conv1_2 = self.conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0, fixed = fixed)
        self.net[name] = L.Eltwise(conv3, conv1_2, operation=P.Eltwise.SUM)
        self.net[name + "_relu"] = L.ReLU(self.net[name], in_place=True)
        return self.net[name + "_relu"]

    def residual_block_shortcut_basic(self, name, bottom, num_filter, stride=1, fixed = False):
        conv1 = self.conv_factory(name + "_branch2b", bottom, 3, num_filter, stride, 1, fixed = fixed)
        conv2 = self.conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, stride, 1, fixed = fixed)
        self.net[name] = L.Eltwise(bottom, conv2, name = name, operation=P.Eltwise.SUM)
        self.net[name + "_relu"] = L.ReLU(self.net[name], name = name + "_relu" , in_place=True)
        return self.net[name + "_relu"]
        
    def residual_block_basic(self, name, bottom, num_filter, stride=1, fixed = False):
        conv1 = self.conv_factory(name + "_branch2b",bottom, 3, num_filter, 1, 1, fixed = fixed)
        conv2 = self.conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, 1, 0, fixed = fixed)
        conv1_2 = self.conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0, fixed = fixed)
        self.net[name] = L.Eltwise(conv2, conv1_2, operation=P.Eltwise.SUM)
        self.net[name + "_relu"] = L.ReLU(self.net[name], name = name + "_relu" , in_place=True)
        return self.net[name + "_relu"]

    def resnet_rcnn(self):
        channals = self.channals
        if not self.deploy:
            data, im_info, gt_boxes = self.data_layer_train()
        else:
            data, im_info = self.data_layer_test()
            gt_boxes = None
        conv1 = self.conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True)
        pool1 = self.pooling_layer(3, 2, 'MAX', 'pool1', conv1)
        k=0
        index = 1
        out = pool1
        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j==0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2  
                    if self.module == "normal":
                        out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                    else:
                        out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    if self.module == "normal":
                        out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                    else:
                        out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)
            channals *= 2

        if not self.deploy:
            rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data, fixed=True)
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois, scores = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        
        feat_aligned = self.roi_align(out, rois)
        out = feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j==0:
                if index == 2:
                    stride = 1
                else:
                    stride = 2
                if self.module == "normal":
                    out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                if self.module == "normal":
                    out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                else:
                    out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)
        pool5 = self.ave_pool(7, 1, "pool5", out)
        cls_score, bbox_pred = self.final_cls_bbox(pool5)

        if not self.deploy:
            self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight= 1, propagate_down=[1,0])
            self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,\
                                loss_weight= 1)
        else:
            self.net["cls_prob"] =  L.Softmax(cls_score)
        return self.net.to_proto()

    def resnet_mask_rcnn(self):
        channals = self.channals
        if not self.deploy:
            data, im_info, gt_boxes, ins_crop = self.data_layer_train_with_ins()
        else:
            data, im_info = self.data_layer_test()
            gt_boxes = None
        conv1 = self.conv_factory("conv1", data, 7, channals, 2, 3, bias_term=True)
        pool1 = self.pooling_layer(3, 2, 'MAX', 'pool1', conv1)
        k=0
        index = 1
        out = pool1
        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j==0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2  
                    if self.module == "normal":
                        out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                    else:
                        out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    if self.module == "normal":
                        out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                    else:
                        out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)
            channals *= 2

        if not self.deploy:
            rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)
            self.net["ins_crop"] = L.Python(rois, ins,
                            name = 'ins_crop',
                            python_param=dict(
                                            module='crop_seg.layer',
                                            layer='CropSegLayer',
                                            param_str='"pooled_w": %s \n "pooled_w": %s ' %(self.pooled_w, self.pooled_w)
                                            ),
                            ntop=1,)
            ins_crop = self.net["ins_crop"]
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        
        feat_aligned = self.roi_align(out, rois)
        out = feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j==0:
                if index == 2:
                    stride = 1
                else:
                    stride = 2
                if self.module == "normal":
                    out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                if self.module == "normal":
                    out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                else:
                    out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)
        
        # for bbox detection
        pool5 = self.ave_pool(7, 1, "pool5", out)
        cls_score, bbox_pred = self.final_cls_bbox(pool5)

        if not self.deploy:
            self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight= 1, propagate_down=[1,0])
            self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,\
                                loss_weight= 1)
        else:
            self.net["cls_prob"] =  L.Softmax(cls_score)
        
        #for mask prediction
        mask_conv1 = self.conv_factory("mask_conv1", feat_aligned, 3, 256, 1, 1, bias_term=True)
        mask_out = self.conv_factory("mask_out", mask_conv1, 1, self.classes, 1, 0, bias_term=True)
        if not self.deploy:
            self.net["loss_mask"] = L.SoftmaxWithLoss(mask_out, ins_crop, loss_weight= 1, propagate_down=[1,0],
                                                    loss_param = dict(
                                                    normalization = 1
                                                    ))
        else:
            self.net["mask_prob"] =  L.Softmax(cls_score)

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
        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    if self.module == "normal":
                        out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, \
                                                  fixed=pre_traned_fixed)
                    else:
                        out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride, \
                                                        fixed=pre_traned_fixed)
                else:
                    if self.module == "normal":
                        out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals, \
                                                           fixed=pre_traned_fixed)
                    else:
                        out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals, \
                                                                 fixed=pre_traned_fixed)
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
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    if self.module == "normal":
                        out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                    else:
                        out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    if self.module == "normal":
                        out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                    else:
                        out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)
            if stage==1:
                self.net["silence_res"] = L.Silence(out, ntop=0)

            if stage==2:
                # for bbox detection
                pool5 = self.ave_pool(7, 1, "pool5", out)
                cls_score, bbox_pred = self.final_cls_bbox(pool5)
                self.net["silence_cls_score"] = L.Silence(cls_score, ntop=0)
                self.net["silence_bbox_pred"] = L.Silence(bbox_pred, ntop=0)

                # for mask prediction
                mask_conv1 = self.conv_factory("mask_conv1", self.net["dummy_roi_pool_conv5"], 3, 256, 1, 1, bias_term=True)
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
        for i in self.stages[:-1]:
            index += 1
            for j in range(i):
                if j == 0:
                    if index == 2:
                        stride = 1
                    else:
                        stride = 2
                    if self.module == "normal":
                        out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
                    else:
                        out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride, fixed=pre_traned_fixed)
                else:
                    if self.module == "normal":
                        out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
                    else:
                        out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals, fixed=pre_traned_fixed)
            channals *= 2

        if not self.deploy:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data, fixed=True)
            self.net["silence_rpn_cls_score_reshape"] = L.Silence(rpn_cls_score_reshape, ntop=0)
            self.net["silence_rpn_bbox_pred"] = L.Silence(rpn_bbox_pred, ntop=0)
        else:
            rpn_cls_score_reshape, rpn_bbox_pred = self.rpn(out, gt_boxes, im_info, data)
            rois, scores = self.roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes)

        det_feat_aligned = self.roi_align("det", out, rois)
        out = det_feat_aligned

        if not self.deploy:
            mask_feat_aligned = self.roi_align("mask", out, mask_rois)
        else:
            mask_feat_aligned = det_feat_aligned

        index += 1
        for j in range(self.stages[-1]):
            if j == 0:
                if index == 2:
                    stride = 1
                else:
                    stride = 2
                if self.module == "normal":
                    out = self.residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride)
                else:
                    out = self.residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride)
            else:
                if self.module == "normal":
                    out = self.residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals)
                else:
                    out = self.residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals)

            # for bbox detection
        pool5 = self.ave_pool(7, 1, "pool5", out)
        cls_score, bbox_pred = self.final_cls_bbox(pool5)

        if not self.deploy:
            self.net["loss_cls"] = L.SoftmaxWithLoss(cls_score, labels, loss_weight=1, propagate_down=[1, 0])
            self.net["loss_bbox"] = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
                                                   loss_weight=1)
        else:
            self.net["cls_prob"] = L.Softmax(cls_score)

        # for mask prediction
        mask_conv1 = self.conv_factory("mask_conv1", mask_feat_aligned, 3, 256, 1, 1, bias_term=True)
        mask_out = self.conv_factory("mask_out", mask_conv1, 1, self.classes, 1, 0, bias_term=True)
        if not self.deploy:
            self.net["loss_mask"] = L.SoftmaxWithLoss(mask_out, masks, loss_weight=1, propagate_down=[1, 0],
                                                      loss_param=dict(
                                                          normalization=1,
                                                          ignore_label = -1
                                                      ))
        else:
            self.net["mask_prob"] = L.Softmax(cls_score)

        return self.net.to_proto()

def main():
    resnet_test = ResNet(deploy=True)
    resnet_train = ResNet(deploy=False)
    #for net in ('18', '34', '50', '101', '152'):
    with open('test.prototxt', 'w') as f:
        f.write(str(resnet_test.resnet_mask_rcnn_rpn()))
    with open('train.prototxt', 'w') as f:
        f.write(str(resnet_train.resnet_mask_rcnn_rpn(stage=1)))

if __name__ == '__main__':
    main()
