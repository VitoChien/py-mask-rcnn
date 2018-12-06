from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections
import caffe
from caffe import layers as L, params as P, to_proto

def roi_align(net, name, bottom, roi, pooling, pooled_w, pooled_h, feat_stride):
    if pooling == "align":
        net["ROIAlign" + name] = L.ROIAlign(bottom, roi, roi_align_param={
            "pooled_w": pooled_w,
            "pooled_h": pooled_h,
            "spatial_scale": 1 / float(feat_stride)})
        return net["ROIAlign" + name]
    else:
        net["ROIPooling"] = L.ROIPooling(bottom, roi, roi_pooling_param={
            "pooled_w": pooled_w,
            "pooled_h": pooled_h,
            "spatial_scale": 1 / float(feat_stride)})
        return net["ROIPooling"]


def conv_factory(net, name, bottom, ks, nout, stride=1, pad=0, bias_term=False, fixed=False, param=None, deploy=False):
    if param == None:
        if not fixed:
            net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                           num_output=nout, pad=pad, bias_term=bias_term,
                                           weight_filler=dict(type='msra'), engine=2)
        else:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'lr_mult': 0, 'decay_mult': 0}, {'lr_mult': 0, 'decay_mult': 0}],
                                               engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'lr_mult': 0, 'decay_mult': 0}], engine=2)
    else:
        if not fixed:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w"}, {'name': param + "_b"}], engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w"}], engine=2)
        else:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w", 'lr_mult': 0, 'decay_mult': 0},
                                                      {'name': param + "_b", 'lr_mult': 0, 'decay_mult': 0}],
                                               engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w", 'lr_mult': 0}], engine=2)

    if "res" in name:
        net[name.replace("res", "bn")] = L.BatchNorm(net[name], in_place=True,
                                                          batch_norm_param=dict(use_global_stats=deploy))
        net[name.replace("res", "scale")] = L.Scale(net[name.replace("res", "bn")], in_place=True,
                                                         scale_param=dict(bias_term=True))
        net[name + "_relu"] = L.ReLU(net[name.replace("res", "scale")], in_place=True)
    else:
        net["bn_" + name] = L.BatchNorm(net[name], in_place=True,
                                             batch_norm_param=dict(use_global_stats=deploy))
        net["scale_" + name] = L.Scale(net["bn_" + name], in_place=True, scale_param=dict(bias_term=True))
        net[name + "_relu"] = L.ReLU(net["scale_" + name], in_place=True)
    return net[name + "_relu"]


def conv_factory_inverse_no_relu(net, name, bottom, ks, nout, stride=1, pad=0, bias_term=False, fixed=False,
                                 param=False, deploy=False):
    if not param:
        if not fixed:
            net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                           num_output=nout, pad=pad, bias_term=bias_term,
                                           weight_filler=dict(type='msra'), engine=2)
        else:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'lr_mult': 0, 'decay_mult': 0}, {'lr_mult': 0, 'decay_mult': 0}],
                                               engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'lr_mult': 0, 'decay_mult': 0}], engine=2)
    else:
        if not fixed:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w"}, {'name': param + "_b"}], engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w"}], engine=2)
        else:
            if bias_term:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w", 'lr_mult': 0, 'decay_mult': 0},
                                                      {'name': param + "_b", 'lr_mult': 0, 'decay_mult': 0}], \
                                               engine=2)
            else:
                net[name] = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                               num_output=nout, pad=pad, bias_term=bias_term,
                                               weight_filler=dict(type='msra'),
                                               param=[{'name': param + "_w", 'lr_mult': 0}], engine=2)

    if "res" in name:
        net[name.replace("res", "bn")] = L.BatchNorm(net[name], in_place=True,
                                                          batch_norm_param=dict(use_global_stats=deploy))
        net[name.replace("res", "scale")] = L.Scale(net[name.replace("res", "bn")], in_place=True,
                                                         scale_param=dict(bias_term=True))
        return net[name.replace("res", "scale")]
    else:
        net["bn_" + name] = L.BatchNorm(net[name], in_place=True,
                                             batch_norm_param=dict(use_global_stats=deploy))
        net["scale_" + name] = L.Scale(net["bn_" + name], in_place=True, scale_param=dict(bias_term=True))
        return net["scale_" + name]


def rpn(net, bottom, gt_boxes, im_info, data, anchors, feat_stride, scales, fixed=False, deploy=False):
    if not fixed:
        net["rpn_conv/3x3"] = L.Convolution(bottom, kernel_size=3, stride=1,
                                                 num_output=512, pad=1,
                                                 param=[{'lr_mult': 1}, {'lr_mult': 2}],
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0), engine=2)
    else:
        net["rpn_conv/3x3"] = L.Convolution(bottom, kernel_size=3, stride=1,
                                                 num_output=512, pad=1,
                                                 param=[{'lr_mult': 0}, {'lr_mult': 0}],
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0), engine=2)
    net["rpn_relu/3x3"] = L.ReLU(net["rpn_conv/3x3"], in_place=True)
    if not fixed:
        net["rpn_cls_score"] = L.Convolution(net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                                  num_output=2 * anchors, pad=0,
                                                  param=[{'lr_mult': 1}, {'lr_mult': 2}],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0), engine=2)
        net["rpn_bbox_pred"] = L.Convolution(net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                                  num_output=4 * anchors, pad=0,
                                                  param=[{'lr_mult': 1}, {'lr_mult': 2}],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0), engine=2)
    else:
        net["rpn_cls_score"] = L.Convolution(net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                                  num_output=2 * anchors, pad=0,
                                                  param=[{'lr_mult': 0}, {'lr_mult': 0}],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0), engine=2)
        net["rpn_bbox_pred"] = L.Convolution(net["rpn_relu/3x3"], kernel_size=1, stride=1,
                                                  num_output=4 * anchors, pad=0,
                                                  param=[{'lr_mult': 0}, {'lr_mult': 0}],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant', value=0), engine=2)
    net["rpn_cls_score_reshape"] = L.Reshape(net["rpn_cls_score"],
                                                  reshape_param={"shape": {"dim": [0, 2, -1, 0]}})

    if (not deploy) and (not fixed):
        net["rpn_labels"], net["rpn_bbox_targets"], net["rpn_bbox_inside_weights"], net[
            "rpn_bbox_outside_weights"] = \
            L.Python(net["rpn_cls_score"], gt_boxes, im_info, data,
                     name='rpn-data',
                     python_param=dict(
                         module='rpn.anchor_target_layer',
                         layer='AnchorTargetLayer',
                         param_str='{"feat_stride": %s,"scales": %s}' % (feat_stride, scales)),
                     # param_str='"feat_stride": %s \n "scales": !!python/tuple %s ' %(feat_stride, scales)),
                     ntop=4, )
        net["rpn_cls_loss"] = L.SoftmaxWithLoss(net["rpn_cls_score_reshape"], net["rpn_labels"],
                                                     name="rpn_loss_cls", propagate_down=[1, 0], \
                                                     loss_weight=1, loss_param={"ignore_label": -1, "normalize": True})
        net["rpn_loss_bbox"] = L.SmoothL1Loss(net["rpn_bbox_pred"], net["rpn_bbox_targets"], \
                                                   net["rpn_bbox_inside_weights"],
                                                   net["rpn_bbox_outside_weights"], \
                                                   name="loss_bbox", loss_weight=1, smooth_l1_loss_param={"sigma": 3.0})
        return net["rpn_cls_loss"], net["rpn_loss_bbox"], net["rpn_cls_score_reshape"], net[
            "rpn_bbox_pred"]
    else:
        return net["rpn_cls_score_reshape"], net["rpn_bbox_pred"]


def roi_proposals(net, rpn_cls_score_reshape, rpn_bbox_pred, im_info, anchors, feat_stride, scales, classes, out_w, deploy=False):
    net["rpn_cls_prob"] = L.Softmax(rpn_cls_score_reshape, name="rpn_cls_prob")
    net["rpn_cls_prob_reshape"] = L.Reshape(net["rpn_cls_prob"], name="rpn_cls_prob_reshape", \
                                                 reshape_param={"shape": {"dim": [0, 2 * anchors, -1, 0]}})

    if not deploy:
        net["rpn_rois"] = L.Python(net["rpn_cls_prob_reshape"], rpn_bbox_pred, im_info,
                                        name='proposal',
                                        python_param=dict(
                                            module='rpn.proposal_layer',
                                            layer='ProposalLayer',
                                            param_str='{"feat_stride": %s,"scales": %s}' % (
                                            feat_stride, scales)),
                                        # param_str='"feat_stride": %s \n "scales": !!python/tuple %s ' %(feat_stride, scales)),
                                        ntop=1, )
        net["rois"], net["labels"], net["bbox_targets"], net["bbox_inside_weights"], net[
            "bbox_outside_weights"] \
            , net["mask_rois"], net["masks"] = \
            L.Python(net["rpn_rois"], net["gt_boxes"], net["ins"],
                     name='roi-data',
                     python_param=dict(
                         module='rpn.proposal_target_layer',
                         layer='ProposalTargetLayer',
                         param_str='{"num_classes": %s,"out_size": %s}' % (classes, out_w)),
                     ntop=7, )
        return net["rois"], net["labels"], net["bbox_targets"], net["bbox_inside_weights"], \
               net["bbox_outside_weights"], net["mask_rois"], net["masks"]
    else:
        net["rois"], net["scores"] = L.Python(net["rpn_cls_prob_reshape"], rpn_bbox_pred, im_info,
                                                        name='proposal',
                                                        python_param=dict(
                                                            module='rpn.proposal_layer',
                                                            layer='ProposalLayer',
                                                            param_str='{"feat_stride": %s,"scales": %s}' % (
                                                            feat_stride, scales)),
                                                        # param_str='"feat_stride": %s \n "scales": !!python/tuple %s ' %(feat_stride, scales)),
                                                        ntop=2, )
        return net["rois"], net["scores"]


def final_cls_bbox(net, bottom, classes, fixed=False):
    if not fixed:
        net["cls_score"] = L.InnerProduct(bottom, name="cls_score",
                                               num_output=classes,
                                               param=[{'lr_mult': 1}, {'lr_mult': 2}],
                                               weight_filler=dict(type='gaussian', std=0.001),
                                               bias_filler=dict(type='constant', value=0))
        net["bbox_pred"] = L.InnerProduct(bottom, name="bbox_pred",
                                               num_output=4 * classes,
                                               param=[{'lr_mult': 1}, {'lr_mult': 2}],
                                               weight_filler=dict(type='gaussian', std=0.001),
                                               bias_filler=dict(type='constant', value=0))
    else:
        net["cls_score"] = L.InnerProduct(bottom, name="cls_score",
                                               num_output=classes,
                                               param=[{'lr_mult': 0}, {'lr_mult': 0}],
                                               weight_filler=dict(type='gaussian', std=0.001),
                                               bias_filler=dict(type='constant', value=0))
        net["bbox_pred"] = L.InnerProduct(bottom, name="bbox_pred",
                                               num_output=4 * classes,
                                               param=[{'lr_mult': 0}, {'lr_mult': 0}],
                                               weight_filler=dict(type='gaussian', std=0.001),
                                               bias_filler=dict(type='constant', value=0))
    return net["cls_score"], net["bbox_pred"]


def data_layer_train(net, classes, out_h, with_rpn=True, deploy=False):
    if not deploy:
        if with_rpn:
            net["data"], net["im_info"], net["gt_boxes"] = L.Python(
                name='input-data',
                python_param=dict(
                    module='roi_data_layer.layer',
                    layer='RoIDataLayer',
                    param_str='{"num_classes": %s,"output_h_w": %s}' % (classes, out_h)),
                ntop=3, )
            return net["data"], net["im_info"], net["gt_boxes"]
        else:
            net["data"], net["rois"], net["labels"], net["bbox_targets"], net[
                "bbox_inside_weights"], \
            net["bbox_outside_weights"] = L.Python(
                name='input-data',
                python_param=dict(
                    module='roi_data_layer.layer',
                    layer='RoIDataLayer',
                    param_str='{"num_classes": %s,"output_h_w": %s}' % (classes, out_h)),
                ntop=3, )
            return net["data"], net["rois"], net["labels"], net["bbox_targets"], net[
                "bbox_inside_weights"], \
                   net["bbox_outside_weights"]


def data_layer_test(net, with_roi=False, deploy=False):
    if deploy:
        if not with_roi:
            net["data"] = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
            net["im_info"] = L.Input(shape=[dict(dim=[1, 3])])
            return net["data"], net["im_info"]
        else:
            net["data"] = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
            net["rois"] = L.Input(shape=[dict(dim=[1, 4])])
            return net["data"], net["rois"]


def data_layer_train_with_ins(net, classes, with_rpn=False, deploy=False):
    if not deploy:
        if with_rpn:
            # net["data"], net["im_info"], net["gt_boxes"], net["mask_rois"], net["masks"]= L.Python(
            #                     name = 'input-data',
            #                     python_param=dict(
            #                                     module='roi_data_layer.layer',
            #                                     layer='RoIDataLayer',
            #                                     param_str='{"num_classes": %s,"output_h_w": %s}' %(classes, out_h)),
            #                     ntop=5,)
            # return net["data"], net["im_info"], net["gt_boxes"], net["mask_rois"], net["masks"]
            net["data"], net["im_info"], net["gt_boxes"], net["ins"] = L.Python(
                name='input-data',
                python_param=dict(
                    module='roi_data_layer_with_instance.layer',
                    layer='RoIDataLayer',
                    param_str='{"num_classes": %s}' % (classes)),
                ntop=4, )
            return net["data"], net["im_info"], net["gt_boxes"], net["ins"]
        else:
            # net["data"], net["rois"], net["labels"], net["bbox_targets"], net["bbox_inside_weights"], \
            # net["bbox_outside_weights"], net["mask_rois"], net["masks"] = L.Python(
            #                     name = 'input-data',
            #                     python_param=dict(
            #                                     module='roi_data_layer.layer',
            #                                     layer='RoIDataLayer',
            #                                     param_str='{"num_classes": %s,"output_h_w": %s}' %(classes, out_h)),
            #                     ntop=8,)
            # return net["data"], net["rois"], net["labels"], net["bbox_targets"], net["bbox_inside_weights"], \
            #         net["bbox_outside_weights"], net["mask_rois"], net["masks"]
            net["data"], net["rois"], net["labels"], net["bbox_targets"], net[
                "bbox_inside_weights"], \
            net["bbox_outside_weights"], net["ins"] = L.Python(
                name='input-data',
                python_param=dict(
                    module='roi_data_layer_with_instance.layer',
                    layer='RoIDataLayer',
                    param_str='{"num_classes": %s}' % (classes)),
                ntop=7, )
            return net["data"], net["rois"], net["labels"], net["bbox_targets"], net[
                "bbox_inside_weights"], \
                   net["bbox_outside_weights"], net["ins"]


def pooling_layer(net, kernel_size, stride, pool_type, layer_name, bottom):
    net[layer_name] = L.Pooling(bottom, pool=eval("P.Pooling." + pool_type), kernel_size=kernel_size,
                                     stride=stride)
    return net[layer_name]


def ave_pool(net, kernel_size, stride, layer_name, bottom):
    return pooling_layer(net, kernel_size, stride, 'AVE', layer_name, bottom)


def residual_block_shortcut(net, name, bottom, num_filter, stride=1, fixed=False, param=None):
    if param != None:
        conv1 = conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, fixed=fixed,
                                  param=param + "_branch2a")
        conv2 = conv_factory(name + "_branch2b", conv1, 3, num_filter, stride, 1, fixed=fixed,
                                  param=param + "_branch2b")
        conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, stride, 0, fixed=fixed,
                                                  param=param + "_branch2c")
    else:
        conv1 = conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, fixed=fixed, param=param)
        conv2 = conv_factory(name + "_branch2b", conv1, 3, num_filter, stride, 1, fixed=fixed, param=param)
        conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, stride, 0, fixed=fixed,
                                                  param=param)
    net[name] = L.Eltwise(bottom, conv3, operation=P.Eltwise.SUM)
    net[name + "_relu"] = L.ReLU(net[name], in_place=True)
    return net[name + "_relu"]


def residual_block(net, name, bottom, num_filter, stride=1, fixed=False, param=None):
    if param != None:
        conv1 = conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, fixed=fixed,
                                  param=param + "_branch2a")
        conv2 = conv_factory(name + "_branch2b", conv1, 3, num_filter, 1, 1, fixed=fixed,
                                  param=param + "_branch2b")
        conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, 1, 0, fixed=fixed,
                                                  param=param + "_branch2c")
        conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0,
                                                    fixed=fixed, param=param + "_branch1")
    else:
        conv1 = conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, fixed=fixed, param=param)
        conv2 = conv_factory(name + "_branch2b", conv1, 3, num_filter, 1, 1, fixed=fixed, param=param)
        conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, 1, 0, fixed=fixed,
                                                  param=param)
        conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0,
                                                    fixed=fixed, param=param)
    net[name] = L.Eltwise(conv3, conv1_2, operation=P.Eltwise.SUM)
    net[name + "_relu"] = L.ReLU(net[name], in_place=True)
    return net[name + "_relu"]


def residual_block_shortcut_basic(net, name, bottom, num_filter, stride=1, fixed=False, param=None):
    if param != None:
        conv1 = conv_factory(name + "_branch2b", bottom, 3, num_filter, stride, 1, fixed=fixed,
                                  param=param + "_branch2b")
        conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, stride, 1, fixed=fixed,
                                                  param=param + "_branch2c")
    else:
        conv1 = conv_factory(name + "_branch2b", bottom, 3, num_filter, stride, 1, fixed=fixed, param=param)
        conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, stride, 1, fixed=fixed,
                                                  param=param)
    net[name] = L.Eltwise(bottom, conv2, name=name, operation=P.Eltwise.SUM)
    net[name + "_relu"] = L.ReLU(net[name], name=name + "_relu", in_place=True)
    return net[name + "_relu"]


def residual_block_basic(net, name, bottom, num_filter, stride=1, fixed=False, param=None):
    if param != None:
        conv1 = conv_factory(name + "_branch2b", bottom, 3, num_filter, 1, 1, fixed=fixed,
                                  param=param + "_branch2b")
        conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, 1, 0, fixed=fixed,
                                                  param=param + "_branch2c")
        conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0,
                                                    fixed=fixed, param=param + "_branch1")
    else:
        conv1 = conv_factory(name + "_branch2b", bottom, 3, num_filter, 1, 1, fixed=fixed, param=param)
        conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, 1, 0, fixed=fixed,
                                                  param=param)
        conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0,
                                                    fixed=fixed, param=param)
    net[name] = L.Eltwise(conv2, conv1_2, operation=P.Eltwise.SUM)
    net[name + "_relu"] = L.ReLU(net[name], name=name + "_relu", in_place=True)
    return net[name + "_relu"]
