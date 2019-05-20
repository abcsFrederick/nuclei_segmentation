import tensorflow as tf


def _parse_record(example_proto):
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'dataset': tf.FixedLenFeature([], tf.string),
        'identifier': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, features)


def _decode_features(parsed_features):
    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    depth = tf.cast(parsed_features['depth'], tf.int32)

    image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [height, width, depth])

    label = tf.decode_raw(parsed_features['label_raw'], tf.uint8)
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [height, width, 1])

    return image, label


def _random_crop(size, seed=None):
    def f(image, label):
        shape = tf.shape(image)
        row = tf.random_uniform([], minval=0, maxval=shape[0] - size[0],
                                dtype=tf.int32, seed=None)
        column = tf.random_uniform([], minval=0, maxval=shape[1] - size[1],
                                   dtype=tf.int32, seed=None)
        image_begin = tf.stack([row, column, 0])
        image_size = tf.stack([size[0], size[1], 3])
        label_begin = tf.stack([row, column, 0])
        label_size = tf.stack([size[0], size[1], 1])
        image = tf.slice(image, image_begin, image_size)
        label = tf.slice(label, label_begin, label_size)
        return image, label
    return f


def _weight_size(size):
    def f(image, label):
        shape = tf.shape(image)
        v = tf.divide(tf.multiply(tf.gather(shape, 0), tf.gather(shape, 1)),
                      tf.constant(size[0] * size[1]))
        r = tf.random_uniform([])
        return tf.less(r, tf.cast(v, tf.float32))
    return f


def _preprocess(image, label):
    scaled = tf.divide(image, tf.constant(127.5, dtype=tf.float32))
    offset = tf.subtract(scaled, tf.constant(1, dtype=tf.float32))
    label = tf.divide(label, tf.constant(255, dtype=tf.float32))
    return offset, label
