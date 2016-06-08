#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OUTDIR=.../BuildData/XXData

EXAMPLE=$OUTDIR
DATA=$OUTDIR
TOOLS=.../caffe/caffe-master/build/tools

TRAIN_DATA_ROOT=$OUTDIR/image/train/
VAL_DATA_ROOT=$OUTDIR/image/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the validation data is stored."
  exit 1
fi

echo "Creating train lmdb pos1..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $TRAIN_DATA_ROOT \
    $DATA/XX_train_pos1.txt \
    $EXAMPLE/XX_train_lmdb_pos1

echo "Creating train lmdb pos2..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $TRAIN_DATA_ROOT \
    $DATA/XX_train_pos2.txt \
    $EXAMPLE/XX_train_lmdb_pos2

echo "Creating train lmdb neg..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $TRAIN_DATA_ROOT \
    $DATA/XX_train_neg.txt \
    $EXAMPLE/XX_train_lmdb_neg

echo "Creating valrank lmdb pos..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $VAL_DATA_ROOT \
    $DATA/XX_valrank_pos.txt \
    $EXAMPLE/XX_valrank_lmdb_pos

echo "Creating valrank lmdb neg..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $VAL_DATA_ROOT \
    $DATA/XX_valrank_neg.txt \
    $EXAMPLE/XX_valrank_lmdb_neg
	
echo "Creating val lmdb pos1..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $VAL_DATA_ROOT \
    $DATA/XX_val_pos1.txt \
    $EXAMPLE/XX_val_lmdb_pos1

echo "Creating val lmdb pos2..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $VAL_DATA_ROOT \
    $DATA/XX_val_pos2.txt \
    $EXAMPLE/XX_val_lmdb_pos2

echo "Creating val lmdb neg..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=false \
    $VAL_DATA_ROOT \
    $DATA/XX_val_neg.txt \
    $EXAMPLE/XX_val_lmdb_neg

echo "Done."
