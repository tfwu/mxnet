#!/usr/bin/env sh

echo "Create the imagenet rec inputs and then train a cnn model using mxnet ..."

# mxnet
MXNET_ROOT=~/Dev/dmlc/mxnet

# imagenet
IMAGENET_ROOT=/media/tfwu/tfwuData/disk2.0tb/ImageNet

IMAGENET_TRAIN_DATA=$IMAGENET_ROOT/ILSVRC2012_img_train
IMAGENET_SYNSETS_FILE=$IMAGENET_ROOT/ILSVRC2012_devkit_t12/data/meta.mat

IMAGENET_VAL_DATA=$IMAGENET_ROOT/ILSVRC2012_img_val
IMAGENET_VAL_GROUNDTRUTH_FILE=$IMAGENET_ROOT/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt

IMAGENET_TEST_DATA=/media/tfwu/tfwuData/disk2.0tb/ImageNet/ILSVRC2012_img_test

# output
OUTPUT_PATH=~/Results/ImageNet
if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir $OUTPUT_PATH
fi

# configuration
# choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn']
NETWORK=inception-bn

# image list
echo "Create the image list files for imagenet train, validation and test datasets ..."
list_ext=.lst
train_list=$OUTPUT_PATH/train
val_list=$OUTPUT_PATH/val
test_list=$OUTPUT_PATH/test
if [ ! -f "$train_list$list_ext" ]; then
     python $MXNET_ROOT/tools/imagenet_make_list.py $IMAGENET_TRAIN_DATA $train_list $IMAGENET_SYNSETS_FILE
fi
if [ ! -f "$val_list$list_ext" ]; then
    python $MXNET_ROOT/tools/imagenet_make_list.py $IMAGENET_VAL_DATA $val_list $IMAGENET_VAL_GROUNDTRUTH_FILE
fi
if [ ! -f "$test_list$list_ext" ]; then
    python $MXNET_ROOT/tools/imagenet_make_list.py $IMAGENET_TEST_DATA $test_list ''
fi
echo "Create the image list files for imagenet train, validation and test datasets ... Done!"

# image rec
echo "Create the image rec ..."
rec_ext=.rec
train_rec=$OUTPUT_PATH/train
val_rec=$OUTPUT_PATH/val
test_rec=$OUTPUT_PATH/test
newSize=256
if [ ! -f "$train_rec$rec_ext" ]; then
    $MXNET_ROOT/bin/im2rec "$train_list$list_ext" "$IMAGENET_TRAIN_DATA/" "$train_rec$rec_ext" resize=$newSize
fi
if [ ! -f "$val_rec$rec_ext" ]; then
    $MXNET_ROOT/bin/im2rec "$val_list$list_ext" "$IMAGENET_VAL_DATA/" "$val_rec$rec_ext" resize=$newSize
fi
if [ ! -f "$test_rec$rec_ext" ]; then
    $MXNET_ROOT/bin/im2rec "$test_list$list_ext" "$IMAGENET_TEST_DATA/" "$test_rec$rec_ext" resize=$newSize
fi
echo  "Create the image rec ... Done!"

# train a cnn model
echo "Train $NETWORK ..."
cd $MXNET_ROOT/example/image-classification
python train_imagenet.py --network "$NETWORK" --data-dir "$OUTPUT_PATH/" --model-prefix "$NETWORK-ImageNet" --gpus 0,1 
