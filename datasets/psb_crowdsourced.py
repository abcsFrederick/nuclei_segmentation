import itertools
import os
import warnings

from PIL import Image

from . import dataset
from .utilities import _affix_keyfunc, _affix_test


class PSBCrowdsourced(dataset.Dataset):
    subdirectory = 'PSB_2015_ImageSize_400'

    affixes = {
        'image': ('', '.tiff'),
        'label': ('', '.TIF'),
    }

    directories = {
        'image': 'Original_Images',
        'label': 'Nuclei_Segmentation',
    }

    subdirectories = [
        'AutomatedMethodSegmentation',
        'ContributorsLevel_1_AggregationLevel_1_537924',
        'ContributorsLevel_2_AggregationLevel_1_537913',
        'ContributorsLevel_2_AggregationLevel_3_536562',
        'ContributorsLevel_2_AggregationLevel_5_537919',
        'ContributorsLevel_3_AggregationLevel_1_523254',
        'Experts_524886',
        'Pathologists_533188',
    ]

    @property
    def _filenames(self):
        image_path = os.path.join(self.path, self.directories['image'])
        filenames = [('image', d) for d in os.listdir(image_path)]

        label_path = os.path.join(self.path, self.directories['label'])
        for subdirectory in self.subdirectories:
            path = os.path.join(label_path, subdirectory)
            filenames += [(subdirectory, d) for d in os.listdir(path)]

        filenames = [(k, v) for (k, v) in filenames if os.path.splitext(v)[1].lower() in ('.png', '.tiff')]

        def keyfunc(value):
            _, filename = value
            return '_'.join(os.path.splitext(filename)[0].split('_')[:2])

        filenames = sorted(filenames, key=keyfunc)

        for root, group in itertools.groupby(filenames, keyfunc):
            group = list(group)
            if group[0][0] != 'image':
                warnings.warn('no image for group %s' % root)
                continue

            if len(group) < 2:
                warnings.warn('no labels for group %s' % root)
                continue

            while group:
                path = os.path.join(self.path, self.directories['label'],
                                    group[-1][0], group[-1][1])
                if Image.open(path).getextrema()[1]:
                    break
                group.pop()

            if not group:
                warnings.warn('no label values for group %s' % root)
                continue

            image_directory = os.path.join(self.directories['image'],
                                           group[0][1])
            label_directory = os.path.join(self.directories['label'],
                                           group[-1][0], group[-1][1])

            yield root, (image_directory, label_directory)


    def labeled_image(self, image_filename, label_filename):
        image = Image.open(os.path.join(self.path, image_filename))
        label = Image.open(os.path.join(self.path, label_filename))
        return image, label.convert('1').convert('L')
