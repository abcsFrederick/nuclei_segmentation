import itertools
import os
import warnings

from PIL import Image

from . import dataset
from .utilities import _affix_keyfunc, _affix_test


class UCSBBioSegmentation(dataset.Dataset):
    subdirectory = 'UCSB_Bio-Segmentation_Benchmark'

    affixes = {
        'image': ('', '_ccd.tif'),
        'label': ('', '.TIF'),
    }

    @property
    def _filenames(self):
        affixes = self.affixes.values()

        keyfunc = _affix_keyfunc(*affixes)

        directories = {
            'image': 'Breast Cancer Cells',
            'label': 'Breast Cancer Cells GroundTruth',
        }

        filenames = []
        for key in ('image', 'label'):
            filenames += os.listdir(os.path.join(self.path, directories[key]))
        filenames = [fn for fn in filenames if _affix_test(fn, affixes)]

        for root, group in itertools.groupby(sorted(filenames), keyfunc):
            if len(list(group)) != len(self.affixes):
                warnings.warn('inconsistent group %s' % root)
                continue
            pair = []
            for key in ('image', 'label'):
                directory = directories[key]
                prefix, suffix = self.affixes[key]
                pair.append(os.path.join(directory, prefix + root + suffix))
            yield root, tuple(pair)

    def labeled_image(self, image_filename, label_filename):
        image = Image.open(os.path.join(self.path, image_filename))
        label = Image.open(os.path.join(self.path, label_filename))
        return image, label
