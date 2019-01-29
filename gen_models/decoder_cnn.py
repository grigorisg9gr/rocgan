import chainer
from chainer import functions as F
from chainer import links as L

from source.miscs.random_samples import sample_continuous


class Decoder(chainer.Chain):
    def __init__(self, dim_z=128, distribution='normal', layers=None, activs=None, 
                 norms='batch', skip_enc=None):
        super(Decoder, self).__init__()
        if layers is None:
            # # format as in Convolution2D: 
            # # out_channels, in_channels, kernel, stride, pad.
            layers = [(512, 512, 4, 4, 0), (512, 64, 4, 2, 1), 
                      (64, 64, 4, 2, 1), (64, 3, 4, 4, 0)]
        self.n_layers = len(layers)
        if activs is None:
            # # if no activations provided, initialize with relu
            # # apart from last layer (tanh).
            activs = [F.leaky_relu for _ in range(self.n_layers - 1)]
            activs.append(F.tanh)
        if isinstance(norms, str) and norms == 'batch':
            # # set the normalization for each layer to batch norm.
            norms = [L.BatchNormalization for _ in range(self.n_layers - 1)]
        elif norms is None or norms == 'None':
            norms = [None] * (self.n_layers - 1)
        self.norms = norms
        self.layers = layers
        self.activs = activs
        assert len(layers) == len(activs) == len(norms) + 1
        # # add the skip list from encoder (or None for no lateral connections).
        self.skip_enc = skip_enc
        if skip_enc is not None:
            # # we create a list (equal to self.n_layers), where if an element
            # # is positive, e.g. ab, it means that its input has a skip 
            # # connection from the output of the ab^{th} layer of the encoder.
            #
            # # initialize the list to -1. Attention: The s1 is 1-based, i.e.
            # # the s1[0] should not be used.
            s1 = [-1] * (self.n_layers + 1)
            # # loop and add the connections. E.g. if skip_enc=[(4, 1)] it
            # # translates to: the input to decoder's 4th layer (from the left)
            # # will be a concatenation of the 3rd layer's output and the output
            # # of the first layer of the encoder. 
            for l_dec, l_enc in skip_enc:
                s1[l_dec] = l_enc
            self.skip_enc = s1

        self.dim_z = dim_z
        self.distribution = distribution
        with self.init_scope():
            # # instead of hardcoding the layers, loop over the list and define them.
            for layer in range(1, self.n_layers):
                setattr(self, 'tconv{}'.format(layer), L.Deconvolution2D(*layers[layer - 1]))
                if self.norms[layer - 1] is not None:
                    setattr(self, 'norm{}'.format(layer), norms[layer - 1](layers[layer - 1][1]))
            # # set the last layer properties, e.g. no batch norm.
            layer = self.n_layers
            setattr(self, 'tconv{}'.format(layer), L.Deconvolution2D(*layers[layer - 1]))

    def __call__(self, x, skips=None, z=None, save_res=False, add_noise=False, **kwargs):
        if z is None and add_noise:
            z = sample_continuous(self.dim_z, len(x), distribution=self.distribution, xp=self.xp)
        if add_noise:
            # # untested format of noise!!
            h = z if len(z.shape) == 4 else F.reshape(z, (z.shape[0], z.shape[1], 1, 1))
        else:
            h = x

        # # save the outputs of each layer if requested.
        outs = []
        # # loop over the layers and apply convolutions along with the
        # # normalizations per layer.
        for l in range(1, self.n_layers):
            print(h.shape, self.skip_enc[l], skips[self.skip_enc[l]].shape)
            if self.skip_enc is not None and self.skip_enc[l] > 0:
                # # in this case, we concatenate the skip with our representation.
                # # The -1 below, since skips includes a zero-based list of all 
                # # encoder layer outputs, i.e. skips[0] -> output of 1st
                # # encoder layer. To get l^th encoder layer's output -> skips[l-1].
                h = F.concat([skips[self.skip_enc[l] - 1], h])
            h = getattr(self, 'tconv{}'.format(l))(h)
            #print('Debugging, decoder step {}: '.format(l), h.shape)
            if self.norms[l - 1] is not None:
                h = getattr(self, 'norm{}'.format(l))(h)
            h = self.activs[l - 1](h)
            if save_res:
                outs.append(h)
        # # last layer:
        l = self.n_layers
        if self.skip_enc is not None and self.skip_enc[l] > 0:
            h = F.concat([skips[self.skip_enc[l] - 1], h])
        h = getattr(self, 'tconv{}'.format(l))(h)
        #print('Debugging, decoder step {}: '.format(self.n_layers), h.shape)
        if save_res:
            # # even though the outputs of the previous layers are after 
            # # activation, this is the opposite.
            outs.append(h)
        h = self.activs[-1](h)
        if not save_res:
            return h
        else:
            return h, outs

