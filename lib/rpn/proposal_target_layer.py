# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import cv2
import math
DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._mask_h_w = layer_params['out_size']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(cfg.TRAIN.BATCH_SIZE, 5, 1, 1)
        # labels
        top[1].reshape(cfg.TRAIN.BATCH_SIZE, 1, 1, 1)
        # bbox_targets
        top[2].reshape(cfg.TRAIN.BATCH_SIZE, self._num_classes * 4, 1, 1)
        # bbox_inside_weights
        top[3].reshape(cfg.TRAIN.BATCH_SIZE, self._num_classes * 4, 1, 1)
        # bbox_outside_weights
        top[4].reshape(cfg.TRAIN.BATCH_SIZE, self._num_classes * 4, 1, 1)
        # mask rois (idx, x1, y1, x2, y2)
        top[5].reshape(cfg.TRAIN.BATCH_SIZE, 5, 1, 1)
        # mask rois (idx, x1, y1, x2, y2)
        top[6].reshape(cfg.TRAIN.BATCH_SIZE, 1, self._mask_h_w, self._mask_h_w)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        mask_file = bottom[2].data
        gt_boxes = gt_boxes.reshape(gt_boxes.shape[0], gt_boxes.shape[1])
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = np.inf if cfg.TRAIN.BATCH_SIZE == -1 else cfg.TRAIN.BATCH_SIZE
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        # print 'proposal_target_layer:', fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights, mask_rois, masks = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes, mask_file, self._mask_h_w)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        # modified by ywxiong
        rois = rois.reshape((rois.shape[0], rois.shape[1], 1, 1))
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        # modified by ywxiong
        labels = labels.reshape((labels.shape[0], 1, 1, 1))
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        # modified by ywxiong
        bbox_targets = bbox_targets.reshape((bbox_targets.shape[0], bbox_targets.shape[1], 1, 1))
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        # modified by ywxiong
        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        bbox_inside_weights = bbox_inside_weights.reshape((bbox_inside_weights.shape[0], bbox_inside_weights.shape[1], 1, 1))
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        mask_rois = mask_rois.reshape((mask_rois.shape[0], mask_rois.shape[1], 1, 1))
        top[5].reshape(*mask_rois.shape)
        top[5].data[...] = mask_rois

        masks = masks.reshape((masks.shape[0], 1, masks.shape[1], masks.shape[2]))
        top[6].reshape(*masks.shape)
        top[6].data[...] = masks

        # print(masks)
        # print(masks.shape)
        # print(np.unique(masks))
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # print 'proposal_target_layer:', bbox_targets.shape
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    if cfg.TRAIN.AGNOSTIC:
        for ind in inds:
            cls = clss[ind]
            start = 4 * (1 if cls > 0 else 0)
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    else:
        for ind in inds:
            cls = clss[ind]
            start = int(4 * cls)
            end = int(start + 4)
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, mask_file, mask_h_w):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    # fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    fg_rois_per_image = int(fg_rois_per_image)
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_image, replace=True)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    # bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = rois_per_image - fg_rois_per_image
    # bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=True)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print 'proposal_target_layer:', keep_inds
    
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_image:] = 0
    rois = all_rois[keep_inds]
    
    # print 'proposal_target_layer:', rois
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # print 'proposal_target_layer:', bbox_target_data
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    mask_rois, roi_has_mask, masks = _get_mask_rcnn_blobs(rois, mask_file, labels, mask_h_w)
    # input()

    return labels, rois, bbox_targets, bbox_inside_weights, mask_rois, masks

def _get_mask_rcnn_blobs(sampled_boxes, mask_file, labels, mask_h_w):
    M = mask_h_w

    mask_file = mask_file[0][0]
    # get gt bboxes from mask
    # return num_ids * [id, x0, y0, x1, y1]
    boxes_from_masks = get_bboxes_from_mask(mask_file)
    # im = mask_file.copy()
    # im = im*20
    # for bbox in boxes_from_masks:
    #     cv2.rectangle(im, (int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4])), 255, 2)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(im, str(int(bbox[0])), (int(bbox[1]), int(bbox[2])), font, 1, 255, 1)
    # cv2.imwrite("mask_gt.png", im)

    fg_inds = np.where(labels> 0)[0]
    roi_has_mask = labels.copy()
    roi_has_mask[roi_has_mask > 0] = 1
    # print("################")
    # print(mask_file.shape)
    # mask_file = (mask_file[:,:,0] != 0 | mask_file[:,:,1] != 0 | mask_file[:,:,2] != 0)
    # mask_file = (mask_file != [0,0,0])[:,:,0]

    if fg_inds.shape[0] > 0:

        # Class labels for the foreground rois
        masks = np.zeros((fg_inds.shape[0], M, M), dtype=np.int32)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = bbox_overlaps(
            np.ascontiguousarray(rois_fg[:, 1:5], dtype=np.float),
            np.ascontiguousarray(boxes_from_masks[:, 1:5], dtype=np.float)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_bbox_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_bbox_ind = fg_bbox_inds[i]
            boxes_from_masks_now=boxes_from_masks[fg_bbox_ind]
            boxes_from_masks_now = boxes_from_masks_now.astype(np.uint16)
            id_now = boxes_from_masks_now[0]
            roi_fg = rois_fg[i]
            # im = cv2.imread(roidb['image'])
            # cv2.rectangle(im, (boxes_from_masks_now[0], boxes_from_masks_now[1]), (boxes_from_masks_now[2], boxes_from_masks_now[3]), (0, 255, 0), 2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(im, 'boxes_from_masks_now', (boxes_from_masks_now[0], boxes_from_masks_now[1]), font, 2, (255, 255, 255), 2)
            # cv2.imwrite("boxes_from_masks_now.jpg", im)
            # im = cv2.imread(roidb['image'])
            # cv2.rectangle(im, (roi_fg[0], roi_fg[1]), (roi_fg[2], roi_fg[3]), (0, 122, 122), 2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(im, 'roi_fg_now', (roi_fg[0], roi_fg[1]), font, 2, (122, 122, 122), 2)
            # cv2.imwrite("roi_fg_now.jpg", im)
            # im = mask_file.copy()
            # cv2.rectangle(im, (boxes_from_masks_now[0], boxes_from_masks_now[1]), (boxes_from_masks_now[2], boxes_from_masks_now[3]), (0, 255, 0), 2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(im, 'boxes_from_masks_now', (boxes_from_masks_now[0], boxes_from_masks_now[1]), font, 2, (255, 255, 255), 2)
            # cv2.imwrite("boxes_from_masks_now_mask.jpg", im*999)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # im = mask_file.copy()
            # im = im*100
            # cv2.rectangle(im, (roi_fg[1], roi_fg[2]), (roi_fg[3], roi_fg[4]), 0, 3)
            # cv2.putText(im, 'roi_fg_now', (roi_fg[1], roi_fg[2]), font, 2, 123, 2)
            # cv2.rectangle(im, (boxes_from_masks_now[1], boxes_from_masks_now[2]), (boxes_from_masks_now[3], boxes_from_masks_now[4]), 0, 3)
            # cv2.putText(im, 'boxes_from_masks_now', (boxes_from_masks_now[1], boxes_from_masks_now[2]), font, 2, 123, 2)
            # cv2.imwrite("roi_fg_now_mask.jpg", im*20)
            # # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            if id_now == 0:
                mask = -np.ones((M, M))
            else:
                mask = get_mask(mask_file, roi_fg, M, id_now)
                mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            # cv2.imwrite("mask_org.png", im)
            # cv2.imwrite("mask_crop.png", mask*999)
            # print("id_now: {}".format(id_now))
            masks[i, :] = mask
            # input()
            # masks[i, :] = np.reshape(mask, (M, M))
    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(labels == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -np.ones((rois_fg.shape[0], M, M), dtype=np.int32)
        # We label it with class = 0 (background)
        mask_class_labels = np.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1
    return rois_fg, roi_has_mask, masks

def get_bboxes_from_mask(mask_in):
    ids = np.unique(mask_in)
    index = (ids != 0)
    ids = ids[index]
    bboxs = np.zeros((len(ids), 5))
    for i, id in enumerate(ids):
        position_now = (mask_in==id)
        rows = np.any(position_now, axis=0)
        cols = np.any(position_now, axis=1)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bboxs[i, 0] = id
        bboxs[i, 1:] = [rmin, cmin, rmax, cmax]
    return bboxs

def get_mask(mask_in, roi, size, id_now):
    roi = roi[1:]
    # print(roi)
    x_start = int(math.floor(roi[0]))
    x_end = int(math.ceil(roi[2]))
    y_start = int(math.floor(roi[1]))
    y_end = int(math.ceil(roi[3]))

    width = mask_in.shape[0]
    height = mask_in.shape[1]
    x_start = min(max(0,x_start), height)
    x_end = min(max(0,x_end), height)
    y_start = min(max(0,y_start), width)
    y_end = min(max(0,y_end), width)

    if x_start == x_end:
        x_end += 1
    if y_start == y_end:
        y_end += 1
    if x_start == mask_in.shape[0]:
        x_start -= 1
    if y_start == mask_in.shape[1]:
        y_start -= 1

    patch_cropped = mask_in[y_start:y_end, x_start:x_end].copy()

    # print(patch_cropped.shape)
    patch_cropped_temp = np.array(patch_cropped == id_now, dtype=np.int32)
    mask = cv2.resize(patch_cropped_temp, (size,size), interpolation=cv2.INTER_NEAREST)
    # mask = np.array(patch_resized == id_pick, dtype=np.int32)
    return mask