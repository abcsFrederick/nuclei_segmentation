import itertools
import os
import warnings

from . import dataset
from . import utilities

from PIL import Image

import nibabel as nib

import numpy as np


class BNS(dataset.Dataset):
    subdirectory = 'ToAnnotate'
    affixes = {
        'image': ('Slide_', '.png'),
        'label': ('GT_', '.nii.gz'),
    }

    @property
    def _filenames(self):
        keyfunc = utilities._affix_keyfunc(*self.affixes.values())

        filenames = []
        for directory in os.listdir(self.path):
            d = os.path.join(self.path, directory)
            if not os.path.isdir(d):
                continue
            filenames += [os.path.join(directory, fn) for fn in os.listdir(d)]
        filenames.sort(key=keyfunc)

        for root, group in itertools.groupby(filenames, keyfunc):
            if len(list(group)) != len(self.affixes):
                warnings.warn('inconsistent group %s' % root)
                continue
            pair = []
            for key in ('image', 'label'):
                prefix, suffix = self.affixes[key]
                pair.append(prefix + root + suffix)
            yield os.path.split(root)[1], tuple(pair)

    def labeled_image(self, image_filename, label_filename):
        image = Image.open(os.path.join(self.path, image_filename))
        label = nib.load(os.path.join(self.path, label_filename))
        label = np.squeeze(np.asarray(label.dataobj))
        label = (label > 0).astype(np.uint8) * 255
        label = np.flip(np.rot90(label), axis=0)
        label = Image.fromarray(label, 'L')
        return image, label
