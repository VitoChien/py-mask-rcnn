# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from utils.cython_bbox import bbox_overlaps
import math

def get_minibatch(roidb, num_classes, mask_h_w):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Get the input seg image blob, formatted for caffe
    # seg_blob, im_scales = _get_seg_blob(roidb, random_scale_inds)

    # Get the input ins image blob, formatted for caffe

    # add seg_blob
    # blobs = {'data': im_blob, 'seg': seg_blob, 'ins': ins_blob}
    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
        mask_rois_blob = np.zeros((0, 5), dtype=np.float32)
        masks_blob = np.zeros((0, 14, 14), dtype=np.float32)
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, mask_rois, masks \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes, mask_h_w)
            batch_ind_mask =  im_i * np.ones((mask_rois.shape[0], 1))
            mask_rois_blob_this_image = np.hstack((batch_ind_mask, mask_rois))
            mask_rois_blob = np.vstack((mask_rois_blob, mask_rois_blob_this_image))
            masks_blob = np.vstack((masks_blob, masks))

        blobs['mask_rois'] = mask_rois_blob
        blobs['masks'] = masks_blob
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        mask_rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        masks_blob = np.zeros((0, 14, 14), dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, mask_rois, masks \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes, mask_h_w)

            # Add to RoIs blob
            rois = im_rois
            # _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            batch_ind_mask =  im_i * np.ones((mask_rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            mask_rois_blob_this_image = np.hstack((batch_ind_mask, mask_rois))

            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
            mask_rois_blob = np.vstack((mask_rois_blob, mask_rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            masks_blob = np.vstack((masks_blob, masks))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

        blobs['mask_rois'] = mask_rois_blob
        blobs['masks'] = masks_blob

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes, mask_h_w):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    sampled_boxes = rois[keep_inds]
    rois = sampled_boxes

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    mask_rois, roi_has_mask, masks = _get_mask_rcnn_blobs(sampled_boxes, roidb, labels, mask_h_w)
    #mask_rois = mask_rois[np.newaxis, :]
    return labels, overlaps, rois, bbox_targets, bbox_inside_weights, mask_rois, masks

def _get_mask_rcnn_blobs(sampled_boxes, roidb, labels, mask_h_w):
    M = mask_h_w

    mask_file = cv2.imread(roidb["ins"], cv2.IMREAD_GRAYSCALE)
    if roidb['flipped']:
        mask_file = mask_file[:, ::-1]

    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0)
    )[0]
    # print(roidb.keys())

    boxes= [roidb['boxes'][i] for i in polys_gt_inds]
    boxes_from_masks = np.zeros((len(boxes), 4), dtype=np.float32)
    for i in range(len(boxes)):
        boxes_from_masks[i, :] = boxes[i]

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
            np.ascontiguousarray(rois_fg, dtype=np.float),
            np.ascontiguousarray(boxes_from_masks, dtype=np.float)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_bbox_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_bbox_ind = fg_bbox_inds[i]
            boxes_from_masks_now=boxes_from_masks[fg_bbox_ind]
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
            # im = mask_file.copy()
            # cv2.rectangle(im, (roi_fg[0], roi_fg[1]), (roi_fg[2], roi_fg[3]), (0, 122, 122), 2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(im, 'roi_fg_now', (roi_fg[0], roi_fg[1]), font, 2, (122, 122, 122), 2)
            # cv2.imwrite("roi_fg_now_mask.jpg", im*999)
            # # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            mask = get_mask(mask_file, roi_fg, boxes_from_masks_now, M)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            # cv2.imwrite("mask.png", mask*999)
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

def get_mask(mask_in, roi, gt_rois, size):
    x_start = int(math.floor(roi[1]))
    x_end = int(math.ceil(roi[3]))
    y_start = int(math.floor(roi[0]))
    y_end = int(math.ceil(roi[2]))

    width = mask_in.shape[0]
    height = mask_in.shape[1]
    x_start = min(max(0,x_start), width)
    x_end = min(max(0,x_end), width)
    y_start = min(max(0,y_start), height)
    y_end = min(max(0,y_end), height)

    if x_start == x_end:
        x_end += 1
    if y_start == y_end:
        y_end += 1
    if x_start == mask_in.shape[0]:
        x_start -= 1
    if y_start == mask_in.shape[1]:
        y_start -= 1

    patch_cropped = mask_in[x_start:x_end, y_start:y_end].copy()

    x_start_gt = int(math.floor(gt_rois[1]))
    x_end_gt = int(math.ceil(gt_rois[3]))
    y_start_gt = int(math.floor(gt_rois[0]))
    y_end_gt = int(math.ceil(gt_rois[2]))

    gt_patch_cropped = mask_in[x_start_gt:x_end_gt, y_start_gt:y_end_gt].copy()
    # find the main obj by count the number of pixel
    ids = np.unique(gt_patch_cropped)
    size_ = -1
    id_pick = -1
    for id in ids:
        if id == 0:
            continue
        mask = (gt_patch_cropped == id)
        arr_new = gt_patch_cropped[mask]
        if arr_new.size > size_:
            size_ = arr_new.size
            id_pick = id

    patch_cropped_temp = np.array(patch_cropped == id_pick, dtype=np.int32)
    # cv2.imwrite("patch_cropped_temp.jpg", patch_cropped_temp*99)

    mask = cv2.resize(patch_cropped_temp, (size,size), interpolation=cv2.INTER_NEAREST)
    # mask = np.array(patch_resized == id_pick, dtype=np.int32)
    return mask

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

# added for reading seg data
def _get_seg_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        seg = cv2.imread(roidb[i]['seg'])
        seg = seg[:, :, :1]
        if roidb[i]['flipped']:
            seg = seg[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        seg, im_scale = prep_seg_for_blob(seg, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(seg)

    # Create a blob to hold the input images
    blob = seg_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
