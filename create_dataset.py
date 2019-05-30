import math
import random

from tfrecord.writer import LabeledImageTFRecordWriter


def main(datasets, filename, test_filename=None, test_split=None,
         validation_filename=None, validation_split=None, random_seed=None):
    writer = LabeledImageTFRecordWriter(filename)

    indices = set()
    for i, dataset in enumerate(datasets):
        for j in range(len(dataset)):
            indices.add((i, j))

    if test_filename and test_split:
        test_size = math.ceil(test_split*len(indices))
        test_indices = set(random.sample(indices, test_size))
        indices -= test_indices
        test_writer = LabeledImageTFRecordWriter(test_filename)
    else:
        test_indices = []

    if validation_filename and validation_split:
        validation_size = math.ceil(validation_split*len(indices))
        validation_indices = set(random.sample(indices, validation_size))
        validation_writer = LabeledImageTFRecordWriter(validation_filename)
    else:
        validation_indices = []

    for i, dataset in enumerate(datasets):
        name = dataset.__class__.__name__
        for j, values in enumerate(dataset.labeled_images):
            identifier, (image, label) = values
            index = (i, j)
            if index in test_indices:
                w = test_writer
            elif index in validation_indices:
                w = validation_writer
            else:
                w = writer
            w.write(image, label, dataset=dataset.__class__.__name__,
                    identifier=identifier)


if __name__ == '__main__':
    import argparse
    import sys

    from datasets import janowczyk, monuseg, bns, ucsb_biosegmentation, psb_crowdsourced

    classes = [
        janowczyk.Janowczyk,
        monuseg.MoNuSeg,
        bns.BNS,
        ucsb_biosegmentation.UCSBBioSegmentation,
        psb_crowdsourced.PSBCrowdsourced,
    ]

    def argument_name(c):
        return c.__module__.split('.')[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--all')
    parser.add_argument('--test_filename')
    parser.add_argument('--test_split', type=float)
    parser.add_argument('--validation_filename')
    parser.add_argument('--validation_split', type=float)
    parser.add_argument('--random_seed', type=int)

    for class_ in classes:
        parser.add_argument('--' + argument_name(class_))

    arguments = parser.parse_args()

    if arguments.random_seed is not None:
        random.seed(arguments.random_seed)

    datasets = []
    for class_ in classes:
        argument = getattr(arguments, argument_name(class_)) or arguments.all
        if argument:
            datasets.append(class_(argument))

    if not datasets:
        sys.exit(2)

    main(datasets, arguments.filename, arguments.test_filename,
         arguments.test_split, arguments.validation_filename,
         arguments.validation_split)
