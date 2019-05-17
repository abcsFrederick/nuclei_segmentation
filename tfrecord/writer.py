import tensorflow as tf

from PIL import Image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def labeled_image_to_feature(image, label, dataset='', identifier=''):
    if isinstance(image, Image.Image):
        width, height = image.size
        depth = len(image.getbands())
    else:
        height, width = image.shape[:2]
        try:
            depth = image.shape[2]
        except IndexError:
            depth = 1

    return {
        'width': _int64_feature(width),
        'height': _int64_feature(height),
        'depth': _int64_feature(depth),
        'label_raw': _bytes_feature(label.tobytes()),
        'image_raw': _bytes_feature(image.tobytes()),
        'dataset': _bytes_feature(dataset.encode('utf-8')),
        'identifier': _bytes_feature(identifier.encode('utf-8')),
    }


class LabeledImageTFRecordWriter(tf.python_io.TFRecordWriter):
    def write(self, image, label, **kwargs):
        feature = labeled_image_to_feature(image, label, **kwargs)
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        record = example.SerializeToString()
        super(LabeledImageTFRecordWriter, self).write(record)
