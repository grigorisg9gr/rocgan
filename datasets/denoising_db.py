import numpy as np
from PIL import Image
import chainer
import random
from chainer import cuda
import chainercv
import PIL

class DenoisingDb(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, size=64, resize_method='bilinear', augmentation=False,
                 crop_size=64, dequantize=True, seed=9, n_classes=0, corrupt=False, 
                 train=True, type_corr='sparse', mask=[0.5, 1.5]):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.size = size
        self.n_classes = n_classes
        if resize_method == 'bilinear':
            self.resize_method = PIL.Image.BILINEAR
        else:
            raise NotImplementedError
        self.augmentation = augmentation
        self.crop_size = crop_size
        self.dequantize = dequantize
        self.corrupt = corrupt
        self.train = train
        self.type_corr = type_corr
        self.m = mask
        if self.corrupt:
            assert self.type_corr in ['sparse', 'denoise', 'im2im']

    def __len__(self):
        return len(self.base)

    def transform(self, image):
        c, h, w = image.shape
        if c == 1:
            image = np.concatenate([image, image, image], axis=0)

        if self.corrupt and self.train:
            # # add the corruptions in this case.
            if self.type_corr == 'denoise':
                mask0 = np.random.uniform(low=self.m[0], high=self.m[1], size=image.shape)
                mask = mask0.astype(np.int32)
                mask = mask.astype(np.float32)
                # # multiply with a binary mask. (the image values are in [0, 255]).
                image1 = image * mask
                image = np.concatenate([image1, image])
            elif self.type_corr == 'sparse':
                # # in this case, it is a binary mask with 
                # # some black pixels (same in all channels).
                mask0 = np.random.uniform(low=self.m[0], high=self.m[1], size=image.shape[1:])
                # # convert to int and then replicate to channels axis.
                mask = mask0.astype(np.int32)
                mask = np.repeat(mask[np.newaxis], 3, axis=0).astype(np.float32)
                image1 = image * mask
                image = np.concatenate([image1, image])
            elif self.type_corr == 'im2im':
                # # assume the image on top is the 'corrupt' image. 
                image1 = image[:, :h // 2, :]
                image2 = image[:, h // 2:, :]
                image = np.concatenate([image1, image2])
                # # change the height.
                h = h // 2

        short_side = h if h < w else w
        crop_size = min(self.crop_size, short_side)
        if self.augmentation:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        _, h, w = image.shape
        if h != self.size and w != self.size:
            image = chainercv.transforms.resize(image, [self.size, self.size], self.resize_method)
        image = image / 128 - 1.
        if self.dequantize:
            image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        return image, label
