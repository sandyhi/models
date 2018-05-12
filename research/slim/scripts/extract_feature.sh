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
DATASET_DIR=$DATA_DIR/mnist
if [ ! -d $DATASET_DIR ]; then mkdir -p $DATASET_DIR; fi


# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=mnist \
#  --dataset_dir=${DATASET_DIR}

# Run evaluation.
python extract_image_feature.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet
