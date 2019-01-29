import sys
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
from source.links.sn_linear import SNLinear
from source.links.sn_convolution_2d import SNConvolution2D
from functools import partial


class CNNDiscriminator(chainer.Chain):
    def __init__(self, layer_d, n_l=4, use_bn=False, sn=True, n_out=2, 
                 ksizes=None, strides=None, paddings=None, add_last=False):
        """
        Initializes the CNN classifier.
        :param layer_d: List of all layers' inputs/outputs (including the input to 
                        the last output).
        :param n_l:  number of layers (not counting the last).
        :param use_bn: Bool, whether to use batch normalization.
        :param sn: Bool, if True use spectral normalization.
        :param n_out:
        :param ksizes: List of the kernel sizes per layer.
        :param strides: List of the stride per layer.
        :param paddings: List of the padding per layer (int).
        :param add_last: Bool, if True it adds a linear layer in the end.
        """
        super(CNNDiscriminator, self).__init__()
        # # the assertion below to ensure that it includes the input and output size as well.
        assert len(layer_d) == n_l + 2
        w = chainer.initializers.GlorotUniform()
        if sn:
            Conv = SNConvolution2D
            Linear = SNLinear
        else:
            Conv = L.Convolution2D
            Linear = L.Linear
        # # initialize args not provided.
        if ksizes is None:
            ksizes = [4] * n_l
        if strides is None:
            strides = [2] * n_l
        if paddings is None:
            paddings = [1] * n_l
        # # the length of the ksizes and the strides is only for the conv layers, hence
        # # it should be one number short of the layer_d.
        assert len(ksizes) == len(strides) == len(paddings) == n_l
        # # save in self, several useful properties.
        self.n_l = n_l
        self.use_bn = use_bn
        self.n_channels = layer_d
        self.add_last = add_last

        with self.init_scope():
            # # iterate over all layers (till the last) and save in self.
            for l in range(1, n_l + 1):
                # # define the input and the output names.
                ni, no = layer_d[l - 1], layer_d[l]
                conv_i = partial(Conv, initialW=w, ksize=ksizes[l - 1], stride=strides[l - 1], pad=paddings[l - 1])
                # # save the self.layer.
                setattr(self, 'l{}'.format(l), conv_i(ni, no))

            if add_last:
                # # save the last layer.
                ni = layer_d[n_l + 1]
                setattr(self, 'l{}'.format(n_l + 1), Linear(ni, n_out, initialW=w))
                self.n_channels.append(n_out)
            # # add the binary classification layer in the end.
            no = n_out if self.add_last else layer_d[n_l + 1]
            self.lin = L.Linear(no, 1, initialW=w)
            if use_bn:
                bn1 = partial(L.BatchNormalization, use_gamma=True, use_beta=False)
                # # set the batch norm (applied before the first layer conv).
                setattr(self, 'bn{}'.format(1), bn1(layer_d[0]))
                for l in range(2, self.n_l + 1):
                    sz = layer_d[l - 1]
                    # # set the batch norm for the layer.
                    setattr(self, 'bn{}'.format(l), bn1(sz))

    def __call__(self, x, y=None, return_feature=False, repres_l=None, **kwargs):
        activ = F.relu
        h = x
        # # last_l is the last layer to compute in
        # # the loop (can be used in case we want to get
        # # features of a previous layer).
        last_l = (self.n_l + self.add_last) if (repres_l is None or 
                                                not return_feature) else repres_l
        # # loop over the layers and get the layers along with the
        # # normalizations per layer.
        for l in range(1, last_l):
            if self.use_bn:
                h = getattr(self, 'bn{}'.format(l))(h)
            h = activ(getattr(self, 'l{}'.format(l))(h))
        # # last layer (no activation).
        repres = getattr(self, 'l{}'.format(last_l))(h)
        output = self.lin(repres) if last_l == self.n_l else None
        if return_feature:
            return output, repres
        else:
            return output

    def __str__(self, **kwargs):
        m1 = 'Layers: {}.\t Info for channels: {}.'
        str1 = m1.format(self.n_l, self.n_channels)
        return str1

