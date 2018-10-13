# --------------------------------------------------------
# Crop the gt
# Written by Tianrui Hui
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import math

DEBUG = False

ph = 28
pw = 28
pch = 1

class CropSegLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        seg_croped_resized = np.zeros((1, pch, ph, pw))
        top[0].reshape(*seg_croped_resized.shape)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        all_rois = bottom[0].data
        seg_gt = bottom[1].data
        labels = bottom[2].data
        ins_gt = bottom[3].data

        # all_rois = all_rois.reshape(all_rois.shape[0], 5)
        # labels = labels.reshape(labels.shape[0])

        # span the seg_gt's channels to the number of labels
        # ==================================================

        # part_label_num = 1
        # new_map_shape = list(seg_gt.shape)
        # new_map_shape[1] = part_label_num
        # depth_label_map = np.zeros(tuple(new_map_shape))
        # for idx, lb in enumerate(range(1, 21)):
        #     new_channel = seg_gt.copy()
        #     bg_idx = np.where(new_channel == 0)
        #     new_channel[new_channel != lb] = -1
        #     new_channel[bg_idx] = 0
        #     new_channel[new_channel == lb] = 1
        #     depth_label_map[:, idx, :, :] = new_channel

        # print 'depth_label_map.shape ', depth_label_map.shape
        # ==================================================

        # print 'all_rois.shape', all_rois.shape
        # print 'labels.shape', labels.shape
        # print 'labels', labels

        pos_ind = np.where(labels > 0)
        # print 'pos_ind', pos_ind
        all_rois = all_rois[pos_ind]  # only keep rois containing positive examples
        # print 'len(all_rois)', len(all_rois)

        if DEBUG:
            print "==============================\nfrom gt_crop\nnum of rois:"
            print len(all_rois)
            print 'seg_gt shape:', seg_gt.shape

        ins_cropped_resized = np.zeros((len(all_rois), pch, ph, pw), dtype=np.float32)

        for ix, roi in enumerate(all_rois):
            x_start = int(math.floor(roi[1]))
            x_end = int(math.ceil(roi[3]))
            y_start = int(math.floor(roi[2]))
            y_end = int(math.ceil(roi[4]))
            if x_start == x_end:
                x_end += 1
            if y_start == y_end:
                y_end += 1
            if x_start == ins_gt.shape[3]:
                x_start -= 1
            if y_start == ins_gt.shape[2]:
                y_start -= 1

            # print [x_start, y_start, x_end, y_end]
            # print 'seg_gt.shape', seg_gt.shape

            # seg_cropped = depth_label_map[:, :, y_start:y_end, x_start:x_end].copy()

            # use instance labels to find the main person in part gt map
            # seg_cropped = seg_gt[:, :, y_start:y_end, x_start:x_end].copy()
            ins_cropped = ins_gt[:, :, y_start:y_end, x_start:x_end].copy()

            full_roi_num = ins_cropped.size
            ins_cropped_ravel = list(ins_cropped.ravel())
            precision_dict = {k: ins_cropped_ravel.count(k) / (full_roi_num + 0.0) for k in set(ins_cropped_ravel)}

            recall_dict = {}
            for d_i, _ in precision_dict.items():
                in_box_num = ins_cropped[ins_cropped == d_i].size
                full_pic_num = ins_gt[ins_gt == d_i].size
                recall = in_box_num / (full_pic_num + 0.0)
                recall_dict[d_i] = recall
            recall_dict = sorted(recall_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

            # print 'precision_dict', precision_dict
            # print 'recall_dict', recall_dict

            max_label = -1
            precision_threshold = 0.2
            for rd_ind, _ in enumerate(recall_dict):
                person = recall_dict[rd_ind][0]
                p_precision = precision_dict[person]
                if p_precision >= precision_threshold:
                    max_label = person
                    break
                else:
                    continue
            if max_label == -1:
                precision_dict = sorted(precision_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                max_label = precision_dict[0][0]

            # print 'max_label', max_label
            # input()

            not_main_ins_inds = np.where(ins_cropped != max_label)
            main_ins_inds = np.where(ins_cropped == max_label)
            bg_inds = np.where(ins_cropped == 0)
            ins_cropped[not_main_ins_inds] = 255
            ins_cropped[bg_inds] = 0
            ins_cropped[main_ins_inds] = 1

            # print 'seg_cropped.shape', seg_cropped.shape
            # input()

            patch_resized = _patch_resize(ins_cropped[0], 'seg')

            ins_cropped_resized[ix, :, :, :] = patch_resized

            # print 'precision_dict', precision_dict
            # print 'max_label', max_label

            if DEBUG:
                print 'roi:'
                print roi
                print 'ins_gt shape: ', ins_gt.shape
                print 'ins_cropped.shape: ', ins_cropped.shape
                print 'ins_cropped'
                print ins_cropped
                print 'w_:', roi[3]-roi[1]
                print 'h_:', roi[4]-roi[2]
                print 'ins_cropped_resized.shape', ins_cropped_resized.shape
                print 'patch_resized'
                print patch_resized[0]
                print 'patch_resized_dict'
                patch_resized_ravel = list(patch_resized.ravel())
                patch_resized_dict = {k: patch_resized_ravel.count(k) for k in set(patch_resized_ravel)}
                print patch_resized_dict
                input()

            # test crop seg labels
            # patch_resized_ravel = list(patch_resized.ravel())
            # precision_dict = {k: patch_resized_ravel.count(k) for k in set(patch_resized_ravel)}
            # precision_dict = sorted(precision_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            # print precision_dict
            # input()

        top[0].reshape(*ins_cropped_resized.shape)
        top[0].data[...] = ins_cropped_resized

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _patch_resize(patch_cropped, option):
    patch_cropped = patch_cropped.transpose((1, 2, 0))
    patch_cropped = patch_cropped.astype(np.float32, copy=False)
    if option == 'seg':
        target_size = (ph, pw)
        # print 'patch_cropped.shape', patch_cropped.shape
        patch_resized = cv2.resize(patch_cropped, target_size, interpolation=cv2.INTER_NEAREST)
        patch_resized = patch_resized[:, :, np.newaxis]
        # print 'patch_resized.shape', patch_resized.shape
        patch_resized = patch_resized.transpose((2, 0, 1))

    if option == 'kps':
        # target_size = (56,56)
        target_size = (27, 27)
        patch_resized = _kps_resize(patch_cropped, target_size)
    return patch_resized


def _kps_resize(patch_croped,target_size):
    patch_resized_k = np.zeros((target_size[0],target_size[1],17), dtype=float)
    patch_shape = patch_croped.shape
    patch_size_w = patch_shape[0]
    patch_size_h = patch_shape[1]
    patch_w_scale = float(target_size[0]) / float(patch_size_w)
    patch_h_scale = float(target_size[1]) / float(patch_size_h)
    position=np.where(patch_croped!=0)
    for i in range(len(position[2])):
        patch_resized_k[patch_w_scale*position[0][i],patch_h_scale*position[1][i],position[2][i]]=1
    # print 'patch_resized_k:',np.where(patch_resized_k!=0)
    # input()
    # patch_resized_k=_mask_add(patch_resized_k)
    return patch_resized_k
