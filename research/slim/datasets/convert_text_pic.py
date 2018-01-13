# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Downloads and converts MNIST-M data to TFRecords of TF-Example protos.

This module downloads the MNIST-M data, uncompresses it, reads the files
that make up the MNIST-M data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import math

# Dependency imports
import numpy as np
from six.moves import urllib
import tensorflow as tf
import cPickle as pk

from datasets import dataset_utils

_IMAGE_SIZE = 32
_NUM_CHANNELS = 3

# The number of images in the training set.
_NUM_TRAIN_SAMPLES = 59001

# The number of images to be kept from the training set for the validation set.
_NUM_VALIDATION = 1000

# The number of images in the test set.
_NUM_TEST_SAMPLES = 9001

# Seed for repeatability.
_RANDOM_SEED = 0


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_png(self, sess, image_data):
        image = sess.run(
            self._decode_png, feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _convert_dataset(split_name, pic_dir, num_shards, data_set, labels, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'valid'.
      dataset_dir: The directory where the converted datasets are stored.
    """
    print('Converting the {} split.'.format(split_name))


    assert split_name in ['train', 'valid']
    num_per_shard = int(math.ceil(len(data_set) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(data_set))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting text picture %d/%d shard %d' % (
                        i+1, len(data_set), shard_id))
                        sys.stdout.flush()

                    # Read the filename:
                    image_data = tf.gfile.FastGFile(
                        os.path.join(pic_dir, data_set[i]), 'r').read()
                    height, width = image_reader.read_image_dims(sess, image_data)
                    class_id = labels[i]

                    example = dataset_utils.image_to_tfexample(image_data, 'png', height,
                                                               width, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'text_pic_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def run(pkl_file, src_pic_dir, dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    print('Saving results to %s' % dataset_dir)
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    raw_data = pk.load(open(pkl_file, 'rb'))

    # train
    train_pic_set = raw_data['train_data']
    train_labels = raw_data['train_label']
    assert len(train_pic_set) == len(train_labels)
    train_shards = int((len(train_pic_set) - 1) / 1024) + 1
    _convert_dataset("train", src_pic_dir, train_shards, train_pic_set, train_labels, dataset_dir)

    # val
    valid_pic_set = raw_data['valid_data']
    valid_labels = raw_data['valid_label']
    assert len(valid_pic_set) == len(valid_labels)
    val_shards = int((len(valid_pic_set) - 1) / 1024) + 1
    _convert_dataset("valid", src_pic_dir, val_shards, valid_pic_set, valid_labels, dataset_dir)

    print('\nFinished converting the Text-Pic dataset!')