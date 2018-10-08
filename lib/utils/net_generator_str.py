from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from string import ascii_lowercase
import collections

def data_layer(netname, classes):
    data_layer_str = '''name: "%s"
layer
{
    name: 'input-data'
    type: 'Python'
    top: 'data'
    top: 'im_info'
    top: 'gt_boxes'
    python_param {
        module: 'roi_data_layer.layer'
        layer: 'RoIDataLayer'
        param_str: "'num_classes': %s"
    }
}
    ''' %(netname, classes)
    return data_layer_str

def conv_layer(conv_params, name, bottom, top=None, filler="msra"):
    if len(conv_params) == 3:
        conv_params = conv_params + ((conv_params[0] - 1) // 2,)
    kernel_size, num_output, stride, pad = conv_params
    if top is None:
        top = name
    conv_layer_str = ('''layer {{
    bottom: "{bottom}"
    top: "{top}"
    name: "{name}"
    type: "Convolution"
    convolution_param {{
        num_output: {num_output}
        kernel_size: {kernel_size}
        pad: {pad}
        stride: {stride}
        weight_filler {{
            type: "msra"
        }}
        '''\
        + ('''bias_term: false\n''' if USE_BN else
     '''bias_filler {{
            type: "constant"
            value: 0
        }}''') +'''
    }}
}}
''').format(**locals())
    return conv_layer_str

def bn_layer(name, bottom, top):
    bn_layer_str = '''layer {{
    bottom: "{top}"
    top: "{top}"
    name: "bn{name}"
    type: "BatchNorm"
    batch_norm_param {{
        use_global_stats: false
    }}
}}
layer {{
    bottom: "{top}"
    top: "{top}"
    name: "scale{name}"
    type: "Scale"
    scale_param {{
        bias_term: true
    }}
}}
'''.format(**locals())
    return bn_layer_str

def in_place_bn(name, activation):
    return bn_layer(name, activation, activation)

def pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top=None):
    if top is None:
        top = layer_name
    pool_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "Pooling"
    pooling_param {
        kernel_size: %d
        stride: %d
        pool: %s
    }
}
'''%(bottom, top, layer_name, kernel_size, stride, pool_type)
    return pool_layer_str

def ave_pool(kernel_size, stride, layer_name, bottom):
    return pooling_layer(kernel_size, stride, 'AVE', layer_name, bottom, layer_name)

def fc_layer(layer_name, bottom, top, num_output=1000):
    fc_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "InnerProduct"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 1
    }
    inner_product_param {
        num_output: %d
        weight_filler {
            type: "xavier"
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
}
'''%(bottom, top, layer_name, num_output)
    return fc_layer_str

def eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
        eltwise_layer_str = '''layer {
    bottom: "%s"
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "Eltwise"
    eltwise_param {
        operation: %s
    }
}
'''%(bottom_1, bottom_2, top, layer_name, op_type)
        return eltwise_layer_str

def activation_layer(layer_name, bottom, top, act_type="ReLU"):
        act_layer_str = '''layer {
    bottom: "%s"
    top: "%s"
    name: "%s"
    type: "%s"
}
'''%(bottom, top, layer_name, act_type)
        return act_layer_str

def in_place_relu(activation_name):
    return activation_layer(activation_name + '_relu', activation_name, activation_name, act_type='ReLU')

def softmax_loss(bottom):
        softmax_loss_str = '''layer {
    bottom: "%s"
    bottom: "label"
    name: "loss"
    type: "SoftmaxWithLoss"
    top: "loss"
}
layer {
    bottom: "%s"
    bottom: "label"
    top: "acc/top-1"
    name: "acc/top-1"
    type: "Accuracy"
    include {
        phase: TEST
    }
}
layer {
    bottom: "%s"
    bottom: "label"
    top: "acc/top-5"
    name: "acc/top-5"
    type: "Accuracy"
    include {
        phase: TEST
    }
    accuracy_param {
        top_k: 5
    }
}
'''%(bottom, bottom, bottom)
        return softmax_loss_str


def conv1_layers():
    layers = conv_layer((7, 64, 2), 'conv1', 'data')
    if USE_BN:
        layers += in_place_bn('_conv1', 'conv1')
    layers += in_place_relu('conv1') \
        + pooling_layer(3, 2, 'MAX', 'pool1', 'conv1')
    return layers

def normalized_conv_layers(conv_params, level, branch, prev_top, activation=True):
    """conv -> batch_norm -> ReLU"""

    name = '%s_branch%s' % (level, branch)
    activation_name = 'res' + name
    layers = conv_layer(conv_params, activation_name, prev_top)
    if USE_BN:
        layers += in_place_bn(name, activation_name)
    if activation:
        layers += in_place_relu(activation_name)
    return layers, activation_name

def bottleneck_layers(prev_top, level, num_output, shortcut_activation=None, shortcut_str='', shortcut_stride=1):
    """1x1 -> 3x3 -> 1x1"""

    if shortcut_activation is None:
        shortcut_activation = prev_top
    all_layers = shortcut_str if USE_SHORTCUT else ''
    layers, prev_top = normalized_conv_layers((1, num_output, shortcut_stride), level, '2a', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((3, num_output, 1), level, '2b', prev_top)
    all_layers += layers
    layers, prev_top = normalized_conv_layers((1, num_output*4, 1), level, '2c', prev_top, activation=(not USE_SHORTCUT))
    all_layers += layers
    if USE_SHORTCUT:
        final_activation = 'res' + level
        all_layers += eltwise_layer(final_activation, shortcut_activation, prev_top, final_activation) \
            + in_place_relu(final_activation)

    return all_layers, prev_top if not USE_SHORTCUT else final_activation

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
        prev_top,               # Previous activation name
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
    shortcut_str, shortcut_activation = normalized_conv_layers(shortcut_params, '%da'%level, '1', prev_top, activation=False)
    network_str = ''
    if sublevel_naming == 'letters' and num_bottlenecks <= 26:
        sublevel_names = ascii_lowercase[:num_bottlenecks]
    else:
        sublevel_names = ['a'] + ['b' + str(i) for i in range(1, num_bottlenecks)]
    for index, sublevel in enumerate(sublevel_names):
        if index != 0:
            shortcut_activation, shortcut_str = None, ''
            layers, prev_top = make_layers(prev_top, '%d%s'%(level, sublevel), num_output, shortcut_activation, shortcut_str)
        else:
            layers, prev_top = make_layers(prev_top, '%d%s'%(level, sublevel), num_output, shortcut_activation, shortcut_str, shortcut_params[2])
        network_str += layers
    return network_str, prev_top

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

    network_str = data_layer('ResNet-' + variant, classes)
    network_str += conv1_layers()
    prev_top = 'pool1'
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
            network_str += rpn(prev_top, anchors)
            network_str += roi_proposals(feat_stride, classes)
            network_str += roi_align(feat_stride)
            prev_top = 'roi_align'
        layers, prev_top = bottleneck_layer_set(prev_top, level, 16*(2**level), num_bottlenecks,
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