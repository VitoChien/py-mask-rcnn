# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def seg_list_to_blob(segs):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([seg.shape for seg in segs]).max(axis=0)
    num_images = len(segs)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                    dtype=np.uint8)
    for i in xrange(num_images):
        seg = segs[i]
        blob[i, 0:seg.shape[0], 0:seg.shape[1], :] = seg
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def ins_list_to_blob(inss):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([ins.shape for ins in inss]).max(axis=0)
    num_images = len(inss)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                    dtype=np.uint8)
    for i in xrange(num_images):
        ins = inss[i]
        blob[i, 0:ins.shape[0], 0:ins.shape[1], :] = ins
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


def prep_seg_for_blob(seg, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    seg = seg.astype(np.uint8, copy=False)
    # im -= pixel_means
    im_shape = seg.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    seg = cv2.resize(seg, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_NEAREST)
    seg = seg[:, :, np.newaxis]

    return seg, im_scale

def prep_ins_for_blob(ins, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    ins = ins.astype(np.uint8, copy=False)
    # im -= pixel_means
    im_shape = ins.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    ins = cv2.resize(ins, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_NEAREST)
    ins = ins[:, :, np.newaxis]

    return ins, im_scale
