import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf
from meta import Meta

tf.app.flags.DEFINE_string('data_dir', './data',
                           'Directory to SVHN (format 1) folders and write the converted files')
FLAGS = tf.app.flags.FLAGS


class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, result_file):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]

        # Get image index
        index = int(path_to_image_file.split('/')[-1].split('.')[0])
        self._example_pointer += 1

        label_of_digits = result_file[index].strip().split(' ')
        length = len(label_of_digits)

        # for digits: 10 represents no digit, for letters: 0 represents no letter
        digits = [10, 10, 10, 10]
        letters = [0, 0, 0, 0, 0]
        idd = 0
        idl = 0
        for i in range(length):
            if i in [0, 4, 5, 6]:
                digits[idd] = int(label_of_digits[i])    # label 10 is essentially digit zero
                idd += 1
            if i in [1, 2, 3, 7, 8]:
                letters[idl] = int(label_of_digits[i])
                idl += 1

        image = Image.open(path_to_image_file).resize([27, 116])
        image = np.array(image).tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'length': ExampleReader._int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits)),
            'letters': tf.train.Feature(int64_list=tf.train.Int64List(value=letters))
        }))
        return example


def convert_to_tfrecords(path_to_dataset_dir_and_digit_result_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_dataset_dir, path_to_digit_result_file in path_to_dataset_dir_and_digit_result_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.jpg'))
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_dataset_dir)

        digit_result_file = open(path_to_digit_result_file, 'r').readlines()
        example_reader = ExampleReader(path_to_image_files)
        for index, path_to_image_file in enumerate(path_to_image_files):
            print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)

            example = example_reader.read_and_convert(digit_result_file)
            if example is None:
                break

            idx = choose_writer_callback(path_to_tfrecords_files)
            writers[idx].write(example.SerializeToString())
            num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_train_dir = os.path.join(FLAGS.data_dir, 'train')
    path_to_test_dir = os.path.join(FLAGS.data_dir, 'test')
    path_to_extra_dir = os.path.join(FLAGS.data_dir, 'extra')
    path_to_train_digit_result_file = os.path.join(path_to_train_dir, 'result.txt')
    path_to_test_digit_result_file = os.path.join(path_to_test_dir, 'result.txt')
    path_to_extra_digit_result_file = os.path.join(path_to_extra_dir, 'result.txt')

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_dir, path_to_train_digit_result_file)],
                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(path_to_test_dir, path_to_test_digit_result_file)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':
    tf.app.run(main=main)
