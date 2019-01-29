import chainer
from chainer import functions as F
from chainer import links as L

from source.links.sn_convolution_2d import SNConvolution2D
from source.miscs.random_samples import sample_continuous


class Encoder(chainer.Chain):
    def __init__(self, dim_z=128, distribution='normal', layers=None, 
                 activs=None, norms='batch', sn=False):
        super(Encoder, self).__init__()
        if sn:
            Conv = SNConvolution2D
        else:
            Conv = L.Convolution2D
        if layers is None:
            # # format as in Convolution2D: 
            # # out_channels, in_channels, kernel, stride, pad.
            layers = [(3, 64, 4, 4, 0), (64, 64, 4, 2, 1),
                      (64, 512, 4, 2, 1), (512, 512, 4, 4, 0)]
        self.n_layers = len(layers)
        if activs is None:
            # # if no activations provided, initialize with relu.
            activs = [F.leaky_relu for _ in range(self.n_layers)]
        if isinstance(norms, str) and norms == 'batch':
            # # set the normalization for each layer to batch norm.
            norms = [L.BatchNormalization for _ in range(self.n_layers - 1)]
        elif norms is None or norms == 'None':
            norms = [None] * (self.n_layers - 1)
        self.norms = norms
        self.layers = layers
        self.activs = activs
        assert len(layers) == len(activs) == len(norms) + 1
        
        self.dim_z = dim_z
        self.distribution = distribution
        with self.init_scope():
            # # instead of hardcoding the layers, loop over the list and define them.
            for layer in range(1, self.n_layers):
                setattr(self, 'conv{}'.format(layer), Conv(*layers[layer - 1]))
                if self.norms[layer - 1] is not None:
                    setattr(self, 'norm{}'.format(layer), norms[layer - 1](layers[layer - 1][1]))
            # # set the last layer properties, e.g. no batch norm.
            layer = self.n_layers
            setattr(self, 'conv{}'.format(layer), Conv(*layers[layer - 1]))

    def __call__(self, x, z=None, save_res=False, add_noise=False, **kwargs):
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
            h = getattr(self, 'conv{}'.format(l))(h)
            #print('Debugging, encoder step {}: '.format(l), h.shape)
            if self.norms[l - 1] is not None:
                h = getattr(self, 'norm{}'.format(l))(h)
            h = self.activs[l - 1](h)
            if save_res:
                outs.append(h)
        # # last layer:
        h = getattr(self, 'conv{}'.format(self.n_layers))(h)
        #print('Debugging, encoder step {}: '.format(self.n_layers), h.shape)
        if save_res:
            outs.append(h)
        h = self.activs[-1](h)
        if not save_res:
            return h
        else:
            return h, outs
