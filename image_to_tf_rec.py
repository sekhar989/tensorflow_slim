## Program for converting any data set to TF Slim format data set

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
import dataset_utils

# Flags for data input
flags = tf.app.flags
flags.DEFINE_string('image_input', '', 'Path to the Image Dataset')
# # flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

in_dataset_dir = FLAGS.image_input
# # out_dataset_dir = FLAGS.output_path

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(in_dataset_dir):
#   """Returns a list of filenames and inferred class names.
#   Args:
#     dataset_dir: A directory containing a set of subdirectories representing
#       class names. Each subdirectory should contain PNG or JPG encoded images.
#   Returns:
#     A list of image file paths, relative to `dataset_dir` and the list of
#     subdirectories, representing class names.
#   """
    dataset_root = os.path.join(in_dataset_dir)
    directories = []
    class_names = []
    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
            
    return photo_filenames, sorted(class_names)


def _get_dataset_filename(in_dataset_dir, split_name, shard_id):
    output_filename = 'dataset_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(in_dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, in_dataset_dir):
#   """Converts the given filenames to a TFRecord dataset.
#   Args:
#     split_name: The name of the dataset, either 'train' or 'validation'.
#     filenames: A list of absolute paths to png or jpg images.
#     class_names_to_ids: A dictionary from class names (strings) to ids
#       (integers).
#     dataset_dir: The directory where the converted datasets are stored.
#   """
    assert split_name in ['train', 'validation']
    
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    
    with tf.Graph().as_default():
        image_reader = ImageReader()
        
        with tf.Session('') as sess:
            
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    in_dataset_dir, split_name, shard_id)
                
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()
                        
                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        
                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
                        
    sys.stdout.write('\n')
    sys.stdout.flush()



def _dataset_exists(in_dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                in_dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(in_dataset_dir):
#   """Runs the download and conversion operation.
#   Args:
#     dataset_dir: The dataset directory where the dataset is stored.
#   """
    if not tf.gfile.Exists(in_dataset_dir):
        tf.gfile.MakeDirs(in_dataset_dir)
        
    if _dataset_exists(in_dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    
    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    photo_filenames, class_names = _get_filenames_and_classes(in_dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]
    
    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     in_dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     in_dataset_dir)
    
    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, in_dataset_dir)
    
    print('\nFinished converting the dataset!')

if __name__ == '__main__':
    run(in_dataset_dir)