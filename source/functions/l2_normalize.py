from chainer import functions as F


def l2_normalize(v, eps=1e-5, axis=1):
    norm = F.sqrt(F.sum(v ** 2, axis=tuple(axis), keepdims=True))
    return v / F.broadcast_to(eps + norm, v.shape)
