from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections
from caffe import layers as L, params as P
def data_layer(classes):
    data, im_info, gt_boxes = L.Python(
                          name = 'input-data',
                          python_param=dict(
                                          module='roi_data_layer.layer',
                                          layer='RoIDataLayer',
                                          param_str='"num_classes": %s' %(classes)),
                          ntop=3,)
    return data, im_info, gt_boxes

def conv_layer(conv_params, name, bottom, filler="msra"):

    if len(conv_params) == 3:
        conv_params = conv_params + ((conv_params[0] - 1) // 2,)
    kernel_size, num_output, stride, pad = conv_params
    if USE_BN:
        conv_layer =  L.Convolution(bottom, name = name, kernel_size=kernel_size, stride=stride,
                                    num_output=num_output, pad=pad, weight_filler = dict(type=filler), bias_term = False)
    else:
        conv_layer =  L.Convolution(bottom, name = name, kernel_size=kernel_size, stride=stride,
                                    num_output=num_output, pad=pad, weight_filler = dict(type=filler),
                                    bias_filler = dict(type='constant', value= 0))
    return conv_layer

def bn_layer(name, bottom):
    bn = L.BatchNorm(bottom, name = "bn" + name, batch_norm_param=dict(use_global_stats=False))
    scale = L.Scale(bn, name = "scale" + name, scale_param=dict(bias_term=True))
    return scale

def in_place_bn(name, bottom):
    return bn_layer(name, bottom)

def pooling_layer(kernel_size, stride, pool_type, layer_name, bottom):
    pooling = L.Pooling(bottom, name = layer_name, pool=eval("P.Pooling." + pool_type), kernel_size=kernel_size, stride=stride)
    return pooling

def ave_pool(kernel_size, stride, layer_name, bottom):
    return pooling_layer(kernel_size, stride, 'AVE', layer_name, bottom, layer_name)

def fc_layer(layer_name, bottom, num_output=1000):
    fc = L.InnerProduct(bottom, name = layer_name, num_output=num_output,
                        param= [{'lr_mult':1, 'decay_mult': 1},{'lr_mult':2, 'decay_mult': 1}],
                        weight_filler = dict(type="xavier"), bias_filler = dict(type='constant', value= 0))
    return fc

def eltwise_layer(layer_name, bottom_1, bottom_2, op_type="SUM"):
    eltwise = L.Eltwise(bottom_1, bottom_2, name = layer_name, eltwise_param=dict(operation = eval("P.Eltwise." + op_type)))
    return eltwise

def activation_layer(layer_name, bottom, act_type="ReLU"):
    activation = eval("L." + act_type)(bottom, name= layer_name)
    return activation

def in_place_relu(activation_name):
    return activation_layer(activation_name + '_relu', activation_name, act_type='ReLU')

def softmax_loss(bottom, label):
    softmax_loss = L.SoftmaxWithLoss(bottom, label)
    return softmax_loss


def conv1_layers(data):
    out = conv_layer((7, 64, 2), 'conv1', data)
    if USE_BN:
        out = in_place_bn('_conv1', out)
    out = in_place_relu(out)
    out = pooling_layer(3, 2, 'MAX', 'pool1', out)
    return out

def normalized_conv_layers(conv_params, level, branch, bottom, activation=True):
    """conv -> batch_norm -> ReLU"""

    name = '%s_branch%s' % (level, branch)
    activation_name = 'res' + name
    out = conv_layer(conv_params, activation_name, bottom)
    if USE_BN:
        out = in_place_bn(name, out)
    if activation:
        out = in_place_relu(out)
    return out, activation_name

def bottleneck_layers(prev_top, level, num_output, shortcut_activation=None, shortcut_str='', shortcut_stride=1):
    """1x1 -> 3x3 -> 1x1"""

    if shortcut_activation is None:
        shortcut_activation = prev_top
    out = shortcut_str if USE_SHORTCUT else ''
    out, prev_top = normalized_conv_layers((1, num_output, shortcut_stride), level, '2a', prev_top)
    out, prev_top = normalized_conv_layers((3, num_output, 1), level, '2b', out)
    out, prev_top = normalized_conv_layers((1, num_output*4, 1), level, '2c', out, activation=(not USE_SHORTCUT))
    if USE_SHORTCUT:
        final_activation = 'res' + level
        out = eltwise_layer(final_activation, shortcut_activation, out, final_activation)
        out = in_place_relu(out)

    return out, prev_top if not USE_SHORTCUT else final_activation

def stacked_layers(prev_top, level, num_output, shortcut_activation=None, shortcut_str='', shortcut_stride=1):
    """3x3 -> 3x3"""

    if shortcut_activation is None:
        shortcut_activation = prev_top
    all_layers = shortcut_str if USE_SHORTCUT else ''
    layers, prev_top = normalized_conv_layers((3, num_output, shortcut_stride), level, '2a', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((3, num_output, 1), level, '2b', prev_top, activation=(not USE_SHORTCUT))
    all_layers += layers
    if USE_SHORTCUT:
        final_activation = 'res' + level
        all_layers += eltwise_layer(final_activation, shortcut_activation, prev_top, final_activation) \
            + in_place_relu(final_activation)

    return all_layers, prev_top if not USE_SHORTCUT else final_activation

def bottleneck_layer_set(
        out,               # Previous activation name
        level,                  # Level number of this set, used for naming
        num_output,             # "num_output" param for most layers of this set
        num_bottlenecks,        # number of bottleneck sets
        shortcut_params='default',    # Conv params of the shortcut convolution
        sublevel_naming='letters', # Naming scheme of layer sets. MSRA sometimes uses letters sometimes numbers
        make_layers=bottleneck_layers, # Function to make layers with
    ):
    """A set of bottleneck layers, with the first one having an convolution shortcut to accomodate size"""

    if shortcut_params == 'default':
        shortcut_params = (1, num_output*(4 if make_layers is bottleneck_layers else 1), 2, 0)
    shortcut_str, shortcut_activation = normalized_conv_layers(shortcut_params, '%da'%level, '1', out, activation=False)
    if sublevel_naming == 'letters' and num_bottlenecks <= 26:
        sublevel_names = ascii_lowercase[:num_bottlenecks]
    else:
        sublevel_names = ['a'] + ['b' + str(i) for i in range(1, num_bottlenecks)]
    for index, sublevel in enumerate(sublevel_names):
        if index != 0:
            shortcut_activation, shortcut_str = None, ''
            layers, out = make_layers(out, '%d%s'%(level, sublevel), num_output, shortcut_activation, shortcut_str)
        else:
            layers, out = make_layers(out, '%d%s'%(level, sublevel), num_output, shortcut_activation, shortcut_str, shortcut_params[2])
    return out

def rpn(bottom, anchors):
    rpn_str = '''
#========= RPN ============

layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "%s"
  top: "rpn/output"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  convolution_param {
    num_output: %s   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}

layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  convolution_param {
    num_output: %s   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}

layer {
   bottom: "rpn_cls_score"
   top: "rpn_cls_score_reshape"
   name: "rpn_cls_score_reshape"
   type: "Reshape"
   reshape_param { shape { dim: 0 dim: 2 dim: -1 dim: 0 } }
}

layer {
  name: 'rpn-data'
  type: 'Python'
  bottom: 'rpn_cls_score'
  bottom: 'gt_boxes'
  bottom: 'im_info'
  bottom: 'data'
  top: 'rpn_labels'
  top: 'rpn_bbox_targets'
  top: 'rpn_bbox_inside_weights'
  top: 'rpn_bbox_outside_weights'
  python_param {
    module: 'rpn.anchor_target_layer'
    layer: 'AnchorTargetLayer'
    param_str: "'feat_stride': 16"
  }
}

layer {
  name: "rpn_loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "rpn_cls_score_reshape"
  bottom: "rpn_labels"
  propagate_down: 1
  propagate_down: 0
  top: "rpn_cls_loss"
  loss_weight: 1
  loss_param {
    ignore_label: -1
    normalize: true
  }
}

layer {
  name: "rpn_loss_bbox"
  type: "SmoothL1Loss"
  bottom: "rpn_bbox_pred"
  bottom: "rpn_bbox_targets"
  bottom: 'rpn_bbox_inside_weights'
  bottom: 'rpn_bbox_outside_weights'
  top: "rpn_loss_bbox"
  loss_weight: 1
  smooth_l1_loss_param { sigma: 3.0 }
}

    '''%(bottom, 2 * anchors, 4 * anchors)

    return rpn_str

def roi_proposals(feat_stride, classes):
    roi_proposals_str = '''
    # ========= RoI Proposal ============

    layer
    {
        name: "rpn_cls_prob"
        type: "Softmax"
        bottom: "rpn_cls_score_reshape"
        top: "rpn_cls_prob"
    }

    layer
    {
        name: 'rpn_cls_prob_reshape'
        type: 'Reshape'
        bottom: 'rpn_cls_prob'
        top: 'rpn_cls_prob_reshape'
        reshape_param {shape {dim: 0 dim: 18 dim: -1 dim: 0}}
    }

    layer
    {
        name: 'proposal'
        type: 'Python'
        bottom: 'rpn_cls_prob_reshape'
        bottom: 'rpn_bbox_pred'
        bottom: 'im_info'
        top: 'rpn_rois'
        #  top: 'rpn_scores'
        python_param {
            module: 'rpn.proposal_layer'
            layer: 'ProposalLayer'
            param_str: "'feat_stride': %s"
        }
    }

    # layer {
    #  name: 'debug-data'
    #  type: 'Python'
    #  bottom: 'data'
    #  bottom: 'rpn_rois'
    #  bottom: 'rpn_scores'
    #  python_param {
    #    module: 'rpn.debug_layer'
    #    layer: 'RPNDebugLayer'
    #  }
    # }

    layer
    {
        name: 'roi-data'
        type: 'Python'
        bottom: 'rpn_rois'
        bottom: 'gt_boxes'
        top: 'rois'
        top: 'labels'
        top: 'bbox_targets'
        top: 'bbox_inside_weights'
        top: 'bbox_outside_weights'
        python_param {
            module: 'rpn.proposal_target_layer'
            layer: 'ProposalTargetLayer'
            param_str: "'num_classes': %s"
        }
    }
    ''' %(feat_stride, classes)
    return roi_proposals_str

def roi_align(stride, pooled_w=7, pooled_h=7):
    roi_align_str = '''
    layer
    {
        bottom: "res4f"
        bottom: "rois"
        top: "roi_align"
        name: "align"
        type: "ROIAlign"
        roi_pooling_param {
            pooled_w: %s
            pooled_h: %s
            spatial_scale: %s
        }
    }
''' %(pooled_w, pooled_h, 1/stride)
    return roi_align_str

def loss(bottom, classes):
    loss_str = '''
layer {
  name: "cls_score"
  type: "InnerProduct"
  bottom: "%s"
  top: "cls_score"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: %s
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "bbox_pred"
  type: "InnerProduct"
  bottom: "%s"
  top: "bbox_pred"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: %s
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "cls_score"
  bottom: "labels"
  propagate_down: 1
  propagate_down: 0
  top: "loss_cls"
  loss_weight: 1
}
layer {
  name: "loss_bbox"
  type: "SmoothL1Loss"
  bottom: "bbox_pred"
  bottom: "bbox_targets"
  bottom: "bbox_inside_weights"
  bottom: "bbox_outside_weights"
  top: "loss_bbox"
  loss_weight: 1
}
''' %(bottom, classes, bottom, 4*classes)
    return loss_str

def resnet(variant='50', classes = 2, anchors = 9, feat_stride = 16): # Currently supports 50, 101, 152
    Bottlenecks = collections.namedtuple('Bottlenecks', ['level', 'num_bottlenecks', 'sublevel_naming'])
    Bottlenecks.__new__.__defaults__ = ('letters',)
    StackedSets = type('StackedSets', (Bottlenecks,), {}) # Makes copy of Bottlenecks class

    out = data_layer(classes)
    out = conv1_layers(out)
    levels = {
        '18': (
            StackedSets(2, 2),
            StackedSets(3, 2),
            StackedSets(4, 2),
            StackedSets(5, 2),
        ),
        '34': (
        	StackedSets(2, 3),
        	StackedSets(3, 4),
        	StackedSets(4, 6),
        	StackedSets(5, 3),
        ),
        '50': (
            Bottlenecks(2, 3),
            Bottlenecks(3, 4),
            Bottlenecks(4, 6),
            Bottlenecks(5, 3),
        ),
        '101': (
            Bottlenecks(2, 3),
            Bottlenecks(3, 4, 'numbered'),
            Bottlenecks(4, 23, 'numbered'),
            Bottlenecks(5, 3),
        ),
        '152': (
            Bottlenecks(2, 3),
            Bottlenecks(3, 8, 'numbered'),
            Bottlenecks(4, 36, 'numbered'),
            Bottlenecks(5, 3),
        )
    }
    for layer_desc in levels[variant]:
        level, num_bottlenecks, sublevel_naming = layer_desc
        if level == 2:
            shortcut_params = (1, (256 if type(layer_desc) is Bottlenecks else 64), 1, 0)
        else:
            shortcut_params = 'default'

        if level == 5:
            out = rpn(out, anchors)
            out = roi_proposals(feat_stride, classes)
            out = roi_align(feat_stride)
            prev_top = 'roi_align'
        out = bottleneck_layer_set(out, level, 16*(2**level), num_bottlenecks,
            shortcut_params=shortcut_params, sublevel_naming=sublevel_naming,
            make_layers=(bottleneck_layers if type(layer_desc) is Bottlenecks else stacked_layers))

        network_str += layers
    network_str += ave_pool(7, 1, 'pool5', prev_top)
    network_str += loss("pool5", classes)
    return network_str


def main():
    for net in ('18', '34', '50', '101', '152'):
        with open('ResNet_{}_train_val.prototxt'.format(net), 'w') as fp:
            fp.write(resnet(net))

USE_SHORTCUT = True
USE_BN = True

if __name__ == '__main__':
    main()