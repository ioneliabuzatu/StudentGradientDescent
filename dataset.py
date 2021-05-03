import os
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_file_from_google_drive, extract_archive
)


class LayerSegmentation(VisionDataset):
    """
    Dataset for segmentation of layers in the retina.

    Inputs are cross-section scans of the retina
    and are represented by a grayscale pillow image.
    The raw labels are the pixel positions of the boundaries
    between the different labels.
    This dataset arranges these raw labels
    into a segmentation map by default.
    """

    TRAIN_URL = ("1rX_xdvg80wQyVhjX_mFZMpZ2Q16A4xc3", {
        'filename': "DLNN2_2020_A4_public_train.zip",
        'md5': "6297cbb5df09abb4a67d8dab4b60e02a"
    })
    TEST_URL = ("1mSrZu092OAxKJAgWznlwK7TGrWzaZXHE", {
        'filename': "DLNN2_2020_A4_public_test.zip",
        'md5': "ad10cc87b262e1dbcb7241f48f81f67c"
    })

    @staticmethod
    def to_image(x: Image, y: np.ndarray, mask=True):
        """
        Put input image and labels into a single image for visualisation.

        Parameters
        ----------
        x : (H, W) Image
            Pillow image from the dataset.
        y : (H, W) ndarray or (3, W) ndarray
            Numpy array holding the segmentation map
        mask : bool, optional
            Whether or not to mask out the invalid class
            using the alpha channel of the result image.

        Returns
        -------
        rgba : (H, W, 4) Image
            RGBA pillow image that visualises inputs and labels.
        """
        rgba = np.array(x.convert('RGBA'))
        if np.shape(x) == y.shape:
            # segmentation map
            rgba[..., 0][y != 1] = 0
            rgba[..., 1][y != 2] = 0
            rgba[..., 2][y != 3] = 0
            if mask:
                rgba[..., -1][y == 0] = 0
            return Image.fromarray(rgba)
        else:
            # raw labels
            ref = np.arange(rgba.shape[0])[:, None]
            for i, layer in enumerate(y):
                layer = np.nan_to_num(layer)
                line = (ref == layer)
                rgba[..., 0][line] = 255 * (i == 2)
                rgba[..., 1][line] = 255 * (i == 1)
                rgba[..., 2][line] = 255 * (i == 0)
                if mask:
                    rgba[:, layer <= 0, -1] = 0
            return Image.fromarray(rgba)

    def __init__(self, root, train=True, segmentation=True, download=False,
                 transforms=None, transform=None, target_transform=None):
        """
        Parameters
        ----------
        root : str
            The root directory to store the dataset.
        train : bool, optional
            Use the training data when `True`,
            otherwise use the test/validation data.
        segmentation : bool, optional
            Provide segmentation maps as labels when `True`,
            otherwise use the raw labels with pixel indices.
        download : bool, optional
            Download and verify the data files when `True`,
            otherwise use the files without download/verification.
        transform : callable, optional
            Function to transform the input image.
        target_transform : callable, optional
            Function to transform the label array.
        """
        super().__init__(root, transforms=transforms, transform=transform,
                         target_transform=target_transform)

        self.segmentation = segmentation

        if download:
            self.download()

        # put dataset in correct mode
        self._train = train
        _, kwargs = self.TRAIN_URL if train else self.TEST_URL
        self.base_dir = kwargs['filename'].split('.')[0]

        # collect filenames
        base_path = Path(self.root) / self.base_dir
        self.images = list(base_path.absolute().glob('*.png'))

    def __getitem__(self, index: int):
        # read image
        img_path = self.images[index]
        img = Image.open(img_path)
        img_shape = np.shape(img)

        # read label
        target_path = img_path.with_suffix('.npy')
        target = np.load(str(target_path))
        target = self.prepare_target(target, img_shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return self.images.__len__()

    def prepare_target(self, target: np.ndarray, x_shape: tuple) -> np.ndarray:
        """ Prepare the target label for this dataset. """
        if not self.segmentation:
            return np.float32(target)

        # convert layers to segmentation map
        seg_map = np.zeros(x_shape, dtype=np.int)
        mask = np.zeros_like(seg_map)
        ref = np.arange(x_shape[0])[:, None]
        for layer in target:
            layer = np.nan_to_num(layer)
            seg_map[ref < layer] += 1
            mask[:, layer > 0] = 1

        return seg_map + mask

    def download(self):
        """ Download, verify and extract the data. """
        for file_id, kwargs in (self.TRAIN_URL, self.TEST_URL):
            download_file_from_google_drive(file_id, self.root, **kwargs)
            archive = os.path.join(self.root, kwargs['filename'])
            base_dir = archive.rsplit('.', maxsplit=1)[0]
            print("Extracting {} to {}".format(archive, base_dir))
            extract_archive(archive, base_dir)
