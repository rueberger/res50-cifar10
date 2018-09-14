""" Module for data fetch and batch gen
"""

import tarfile

import requests

import tensorflow as tf

from reslab.constants import DATA_PATH, DATA_URL

def pull_data():
    response = requests.get(DATA_URL)
    tar_path = '{}.tar.gz'.format(DATA_PATH)
    with open(tar_path, 'wb') as file_handle:
        file_handle.write(response.content)

    with tarfile.open(tar_path, 'r:gz') as file_handle:
        file_handle.extractall()




def get_input_fn(filenames, batch_size, num_epochs, shuffle=False, augment=False):
    label_bytes = 1
    image_height = 32
    image_width = 32
    image_depth = 3
    image_bytes = image_height * image_width * image_depth
    record_bytes = label_bytes + image_bytes

    def _parse_record(value):
        raw_data = tf.decode_raw(value, tf.uint8)
        raw_data.set_shape([record_bytes])
        label = tf.squeeze(tf.cast(raw_data[:label_bytes], tf.int32), axis=0)
        depth_major = tf.reshape(
            raw_data[label_bytes:],
            [image_depth, image_height, image_width])
        image = tf.transpose(depth_major, [1, 2, 0])
        image = tf.to_float(image) / 255
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.clip_by_value(image, 0.0, 1.0)
        features = {'image': image}
        labels = {'label': label}
        return features, labels

    def _input_fn():
        with tf.device('/cpu:0'):
            dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes)
            dataset = dataset.map(
                map_func=_parse_record,
                num_parallel_calls=4)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=1000)
            dataset = dataset.repeat(num_epochs)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
    return _input_fn
