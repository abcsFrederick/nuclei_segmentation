import os.path


class Dataset(object):
    subdirectory = None

    def __init__(self, path):
        self._path = path

    def __len__(self):
        return len(list(self._filenames))

    @property
    def path(self):
        return os.path.join(self._path, self.subdirectory)

    @property
    def _filenames(self):
        raise NotImplementedError

    @property
    def labeled_image(self):
        raise NotImplementedError

    @property
    def labeled_images(self):
        for identifier, filenames in self._filenames:
            yield identifier, self.labeled_image(*filenames)
