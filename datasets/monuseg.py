import itertools
import os
import warnings
import xml.dom.minidom

from PIL import Image, ImageDraw

from . import dataset
from . import utilities


class MoNuSeg(dataset.Dataset):
    subdirectory = os.path.join('MICCAI18_MoNuSeg', 'MoNuSeg Training Data')
    affixes = {
        'image': ('', '.tif'),
        'label': ('', '.xml'),
    }

    def vertex(self, element):
        return tuple(float(element.getAttribute(a)) for a in ('X', 'Y'))

    def polygon(self, region):
        verticies = region.getElementsByTagName('Vertex')
        return tuple(self.vertex(e) for e in verticies)

    def regions(self, document):
        regions = document.documentElement.getElementsByTagName('Region')
        yield from map(self.polygon, regions)

    def label(self, filename, size):
        mask = Image.new('L', size)
        document = xml.dom.minidom.parse(filename)
        for polygon in self.regions(document):
            ImageDraw.Draw(mask).polygon(polygon, fill=255, outline=255)
        return mask

    def labeled_image(self, image_filename, label_filename):
        image = Image.open(os.path.join(self.path, image_filename))
        label_path = os.path.join(self.path, label_filename)
        return image, self.label(label_path, image.size)

    @property
    def _filenames(self):
        keyfunc = utilities._affix_keyfunc(*self.affixes.values())

        directories = {
            'image': 'Tissue images',
            'label': 'Annotations',
        }

        filenames = []
        for key in ('image', 'label'):
            filenames += os.listdir(os.path.join(self.path, directories[key]))

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
