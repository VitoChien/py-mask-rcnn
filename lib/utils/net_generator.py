from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections
from caffe import layers as L, params as P, to_proto

def data_layer(classes):
    data, im_info, gt_boxes = L.Python(
                          name = 'input-data',
                          python_param=dict(
                                          module='roi_data_layer.layer',
                                          layer='RoIDataLayer',
                                          param_str='"num_classes": %s' %(classes)),
                          ntop=3,)
    return data, im_info, gt_boxes

def pooling_layer(kernel_size, stride, pool_type, layer_name, bottom):
    pooling = L.Pooling(bottom, name = layer_name, pool=eval("P.Pooling." + pool_type), kernel_size=kernel_size, stride=stride)
    return pooling

def ave_pool(kernel_size, stride, layer_name, bottom):
    return pooling_layer(kernel_size, stride, 'AVE', layer_name, bottom)


def softmax_loss(bottom, label):
    softmax_loss = L.SoftmaxWithLoss(bottom, label)
    return softmax_loss

def roi_align(bottom, roi, stride, pooling, pooled_w=7, pooled_h=7):
    if pooling == "align":
        roi_align = L.ROIAlign(bottom, roi, name = "ROIAlign", roi_align_param = {
                "pooled_w": pooled_w,
                "pooled_h": pooled_h,
                "spatial_scale": 1/stride})
    else:
        roi_align = L.ROIPooling(bottom, roi, name = "ROIPooling",roi_pooling_param = {
                "pooled_w": pooled_w,
                "pooled_h": pooled_h,
                "spatial_scale": 1/stride})
    return roi_align

def conv_factory(name, bottom, ks, nout, stride=1, pad=0, deploy=False):
    conv = L.Convolution(bottom, name = name, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    if "res" in name:
        name_new = name.split("_")[1]
        batch_norm = L.BatchNorm(conv, name = "bn_" + name_new, in_place=True, batch_norm_param=dict(use_global_stats=deploy))
        scale = L.Scale(batch_norm, name = "scale_" + name_new, in_place=True, scale_param=dict(bias_term=True))
    else:
        batch_norm = L.BatchNorm(conv, name = "bn_" + name, in_place=True, batch_norm_param=dict(use_global_stats=deploy))
        scale = L.Scale(batch_norm, name = "scale_" + name, in_place=True, scale_param=dict(bias_term=True))
    relu = L.ReLU(scale, name = name + "_relu" , in_place=True)
    return relu

def conv_factory_inverse_no_relu(name, bottom, ks, nout, stride=1, pad=0, deploy=False):
    conv = L.Convolution(bottom, name = name, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, name = "bn_" + name, in_place=True, batch_norm_param=dict(use_global_stats=deploy))
    scale = L.Scale(batch_norm, name = "scale_" + name, in_place=True, scale_param=dict(bias_term=True))
    return scale

def residual_block_shortcut(name, bottom, num_filter, stride=1, deploy=False):
    conv1 = conv_factory(name + "_branch2a", bottom, 1, num_filter, stride, 0, deploy)
    conv2 = conv_factory(name + "_branch2b", conv1, 3, num_filter, stride, 1, deploy)
    conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, stride, 0, deploy)
    addition = L.Eltwise(bottom, conv3, name = name, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, name = name + "_relu" , in_place=True)
    return relu


def residual_block(name, bottom, num_filter, stride=1, deploy=False):
    conv1 = conv_factory(name + "_branch2a",bottom, 1, num_filter, stride, 0, deploy)
    conv2 = conv_factory(name + "_branch2b",conv1, 3, num_filter, 1, 1, deploy)
    conv3 = conv_factory_inverse_no_relu(name + "_branch2c", conv2, 1, 4 * num_filter, 1, 0, deploy)
    conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0, deploy)
    addition = L.Eltwise(conv3, conv1_2, name = name, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, name = name + "_relu" , in_place=True)
    return relu

def residual_block_shortcut_basic(name, bottom, num_filter, stride=1, deploy=False):
    conv1 = conv_factory(name + "_branch2b", bottom, 3, num_filter, stride, 1, deploy)
    conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, stride, 1, deploy)
    addition = L.Eltwise(bottom, conv2, name = name, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, name = name + "_relu" , in_place=True)
    return relu


def residual_block_basic(name, bottom, num_filter, stride=1, deploy=False):
    conv1 = conv_factory(name + "_branch2b",bottom, 3, num_filter, 1, 1, deploy)
    conv2 = conv_factory_inverse_no_relu(name + "_branch2c", conv1, 3, 4 * num_filter, 1, 0, deploy)
    conv1_2 = conv_factory_inverse_no_relu(name + "_branch1", bottom, 1, 4 * num_filter, stride, 0, deploy)
    addition = L.Eltwise(conv2, conv1_2, name = name, operation=P.Eltwise.SUM)
    relu = L.ReLU(addition, name = name + "_relu" , in_place=True)
    return relu


def rpn(bottom, gt_boxes, im_info, data, anchors = 9, deploy=False):
    conv = L.Convolution(bottom, name = "rpn_conv/3x3", kernel_size=3, stride=1,
                                num_output=512, pad=1,
                                param= [{'lr_mult':1},{'lr_mult':2}],
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0))
    relu = L.ReLU(conv, name = "rpn_relu/3x3" , in_place=True)
    rpn_cls_score = L.Convolution(relu, name = "rpn_cls_score", kernel_size=1, stride=1,
                                num_output= 2 * anchors, pad=0,
                                param= [{'lr_mult':1},{'lr_mult':2}],
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0))
    rpn_bbox_pred = L.Convolution(relu, name = "rpn_bbox_pred", kernel_size=1, stride=1,
                                num_output= 4 * anchors, pad=0,
                                param= [{'lr_mult':1},{'lr_mult':2}],
                                weight_filler=dict(type='gaussian', std=0.01),
                                bias_filler=dict(type='constant', value=0))
    rpn_cls_score_reshape = L.Reshape(rpn_cls_score, name = "rpn_cls_score_reshape", reshape_param= {"shape" : { "dim": [0, 2, -1, 0]}})

    
    if not deploy:
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                    L.Python(rpn_cls_score, gt_boxes, im_info, data,
                          name = 'rpn-data',
                          python_param=dict(
                                          module='rpn.anchor_target_layer',
                                          layer='AnchorTargetLayer',
                                          param_str='"feat_stride": %s' %(16)),
                          ntop=4,)
        rpn_cls_loss = L.SoftmaxWithLoss(rpn_cls_score_reshape, rpn_labels, name = "rpn_loss_cls", propagate_down=[1,0],\
                        loss_weight = 1, loss_param = {"ignore_label": -1, "normalize": True})
        rpn_loss_bbox = L.SmoothL1Loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, \
                        name= "loss_bbox", loss_weight = 1, smooth_l1_loss_param = {"sigma": 3.0})
        return rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred
    else:
        return rpn_cls_score_reshape, rpn_bbox_pred
        

def roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes, classes, feat_stride = 16, deploy=False):
    rpn_cls_prob = L.Softmax(rpn_cls_score_reshape, name = "rpn_cls_prob")
    rpn_cls_prob_reshape = L.Reshape(rpn_cls_prob, name = "rpn_cls_prob_reshape", \
                reshape_param= {"shape" : { "dim": [0, 18, -1, 0]}})
    rpn_rois = L.Python(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, 
                    name = 'proposal',
                    python_param=dict(
                                    module='rpn.proposal_layer',
                                    layer='ProposalLayer',
                                    param_str='"feat_stride": %s' %(feat_stride)),
                    ntop=1,)
    
    if not deploy:
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                    L.Python(rpn_rois, gt_boxes,
                        name = 'roi-data',
                        python_param=dict(
                                        module='rpn.proposal_target_layer',
                                        layer='ProposalTargetLayer',
                                        param_str='"num_classes": %s' %(classes)),
                        ntop=5,)
        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
    else:
        return rpn_rois


def final_cls_bbox(bottom, classes):
    cls_score = L.InnerProduct(bottom, name = "cls_score",
                            num_output= classes,
                            param= [{'lr_mult':1},{'lr_mult':2}],
                            weight_filler=dict(type='gaussian', std=0.001),
                            bias_filler=dict(type='constant', value=0))
    bbox_pred = L.InnerProduct(bottom, name = "bbox_pred",
                            num_output= 4 * classes,
                            param= [{'lr_mult':1},{'lr_mult':2}],
                            weight_filler=dict(type='gaussian', std=0.001),
                            bias_filler=dict(type='constant', value=0))
    return cls_score, bbox_pred


def resnet(stages=[3, 4, 6, 3], channals=64, deploy=False, classes = 2, anchors = 9, feat_stride = 16, module = "normal"):
    data, label = L.Data(source="", backend=P.Data.LMDB, batch_size=1, ntop=2,
        transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))
    # out = conv_layer((7, 64, 2), 'conv1', data)
    # out = in_place_bn('_conv1', out)
    # out = in_place_relu('conv1', out)
    out = conv_factory("conv1", data, 7, channals, 2, 3)
    out = pooling_layer(3, 2, 'MAX', 'pool1', out)

    k=0
    index = 1
    for i in stages:
        index += 1
        for j in range(i):
            if j==0:
                if index == 2:
                    stride = 1
                else:
                    stride = 2
                if module == "normal":
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
                else:
                    out = residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
            else:
                if module == "normal":
                    out = residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
                else:
                    out = residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
        channals *= 2
    return to_proto(out)

def resnet_rcnn(stages=[3, 4, 6, 3], channals=64, deploy=False, classes = 2, anchors = 9, feat_stride = 16, pooling = "align", module = "normal"):
    if not deploy:
        data, im_info, gt_boxes = data_layer(classes)
    else:
        data = L.DummyData(shape=[dict(dim=[1, 3, 224, 224])])
        im_info = L.DummyData(shape=[dict(dim=[1, 3])])
        gt_boxes = None
    out = conv_factory("conv1", data, 3, channals, 2, 1)
    out = pooling_layer(3, 2, 'MAX', 'pool1', out)
    k=0
    index = 1
    for i in stages[:-1]:
        index += 1
        for j in range(i):
            if j==0:
                if index == 2:
                    stride = 1
                else:
                    stride = 2  
                if module == "normal":
                    out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
                else:
                    out = residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
            else:
                if module == "normal":
                    out = residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
                else:
                    out = residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
        channals *= 2

    if not deploy:
        rpn_cls_loss, rpn_loss_bbox, rpn_cls_score_reshape, rpn_bbox_pred = rpn(out, gt_boxes, im_info, data, deploy=deploy)
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes, classes, deploy=deploy)
    else:
        rpn_cls_score_reshape, rpn_bbox_pred = rpn(out, gt_boxes, im_info, data, deploy=deploy)
        rois = roi_proposals(rpn_cls_score_reshape, rpn_bbox_pred, im_info, gt_boxes, classes, deploy=deploy)

    
    feat_aligned = roi_align(out, rois, feat_stride, "align")
    out = feat_aligned

    index += 1
    for j in range(stages[-1]):
        if j==0:
            if index == 2:
                stride = 1
            else:
                stride = 2
            if module == "normal":
                out = residual_block("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
            else:
                out = residual_block_basic("res" + str(index) + ascii_lowercase[j], out, channals, stride, deploy=deploy)
        else:
            if module == "normal":
                out = residual_block_shortcut("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
            else:
                out = residual_block_shortcut_basic("res" + str(index) + ascii_lowercase[j], out, channals, deploy=deploy)
    pool5 = ave_pool(7, 1, "pool5", out)
    cls_score, bbox_pred = final_cls_bbox(pool5, classes)

    if not deploy:
        loss_cls = L.SoftmaxWithLoss(cls_score, labels, name = "loss_cls",propagate_down=[1,0], loss_weight= 1)
        loss_bbox = L.SmoothL1Loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights,\
                name = "loss_bbox", propagate_down=[1,0], loss_weight= 1)
        return to_proto(rpn_cls_loss, rpn_loss_bbox, loss_cls, loss_bbox)
    else:
        cls_prob =  L.Softmax(cls_score, name = "cls_prob")
        return to_proto(cls_prob, bbox_pred)

def main():
    #for net in ('18', '34', '50', '101', '152'):
    with open('ResNet_50_deploy.prototxt', 'w') as f:
        f.write(str(resnet_rcnn(deploy=True)))
    with open('ResNet_50_train_val.prototxt', 'w') as f:
        f.write(str(resnet_rcnn()))

if __name__ == '__main__':
    main()