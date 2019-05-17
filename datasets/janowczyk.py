import itertools
import os
import warnings

from PIL import Image

from . import dataset
from . import utilities


class Janowczyk(dataset.Dataset):

    subdirectory = 'nuclei'

    affixes = {
        'image': ('', '_original.tif'),
        'label': ('', '_mask.png'),
    }

    @property
    def _filenames(self):
        keyfunc = utilities._affix_keyfunc(*self.affixes.values())

        filenames = os.listdir(self.path)

        for root, group in itertools.groupby(sorted(filenames), keyfunc):
            if len(list(group)) != len(self.affixes):
                warnings.warn('inconsistent group %s' % root)
                continue
            pair = []
            for key in ('image', 'label'):
                prefix, suffix = self.affixes[key]
                pair.append(prefix + root + suffix)
            yield root, tuple(pair)

    def labeled_image(self, image_filename, label_filename):
        image = Image.open(os.path.join(self.path, image_filename))
        label = Image.open(os.path.join(self.path, label_filename))
        label = label.convert('L')
        return image, label
