# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import math
import sys
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, end_points, fc3 = network_fn(images)

        print  "logits = ", logits
        print " end_points ", end_points
        print " fc3 ", fc3


        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()


        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path


        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            print (" file : " + ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            # get full tensor list
            op = sess.graph.get_operations()
            tensor_list = [m.values() for m in op]
            print ("tensor list : " + str(tensor_list))


            fc3_tensor = sess.graph.get_tensor_by_name('LeNet/fc3/Relu:0')

            num_iter = int(math.ceil(10000 / float(FLAGS.batch_size)))
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))
                step = 0
                while step < num_iter and not coord.should_stop():
                    step += 1
                    logits_feat, fc3_tensor_feat = sess.run([logits, fc3])
                    if step > 5:
                        sys.exit()
                    print(" logits_feat "+ str(logits_feat))
                    print("fc3_tensor_feat" + str(fc3_tensor_feat))

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


        tf.logging.info('Evaluating %s' % checkpoint_path)


if __name__ == '__main__':
    tf.app.run()
