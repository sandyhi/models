#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset
# 2. Trains a LeNet model on the MNIST training set.
# 3. Evaluates the model on the MNIST testing set.
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh
set -e
ROOT=$(dirname $(cd $(dirname $0); pwd))

# data
DATA_DIR=$ROOT/data
if [ ! -d $DATA_DIR ]; then mkdir -p $DATA_DIR; fi

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=$ROOT/model
if [ ! -d $TRAIN_DIR ]; then mkdir -p $TRAIN_DIR; fi

# Where the dataset is saved to.
DATASET_DIR=$DATA_DIR/text_pic
if [ ! -d $DATASET_DIR ]; then mkdir -p $DATASET_DIR; fi

# data parameters
FULL_RAW_DATA_FILE=/$DATA_DIR/in_full_train.data
PATH_RAW_PKL_FILE=$DATA_DIR/raw_pkl_data.pkl
SRC_PIC_DIR=

# create pickle file
python $ROOT/create_pkl_file.py $FULL_RAW_DATA_FILE $PATH_RAW_PKL_FILE

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=text_pic \
  --dataset_dir=${DATASET_DIR} \
  --pkl_file=$PATH_IN_PKL_FILE \
  --src_pic_dir=$SRC_PIC_DIR

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=text_pic \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --max_number_of_steps=20000 \
  --batch_size=50 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate_decay_type=fixed \
  --weight_decay=0

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=text_pic \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet