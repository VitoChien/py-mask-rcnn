#!/bin/bash
# Usage:
# ./experiments/scripts/mask_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/mask_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
ITER=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=150000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/mask_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/test_net_mask.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/mask_rcnn_end2end/test.prototxt \
  --def_mask models/${PT_DIR}/${NET}/mask_rcnn_end2end/test_mask.prototxt \
  --net ./output/mask_rcnn_end2end/${TRAIN_IMDB}/resnet50_mask_rcnn_iter_${ITER}.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/mask_rcnn_end2end.yml 
