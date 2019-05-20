import math
import random

from tfrecord.writer import LabeledImageTFRecordWriter


def main(datasets, filename, validation_filename=None, validation_split=None,
         random_seed=None):
    writer = LabeledImageTFRecordWriter(filename)

    if validation_filename and validation_split:
        indices = []
        for i, dataset in enumerate(datasets):
            for j in range(len(dataset)):
                indices.append((i, j))
        validation_size = math.ceil(validation_split*len(indices))
        validation_indices = random.sample(indices, validation_size)
        validation_writer = LabeledImageTFRecordWriter(validation_filename)
    else:
        validation_indices = []

    for i, dataset in enumerate(datasets):
        name = dataset.__class__.__name__
        for j, values in enumerate(dataset.labeled_images):
            identifier, (image, label) = values
            if (i, j) in validation_indices:
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

    main(datasets, arguments.filename, arguments.validation_filename,
         arguments.validation_split)
