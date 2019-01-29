import copy
import chainer
from chainer import functions as F
from chainer import cuda


def orth_loss(model):
    params = [
        param for _, param in sorted(model.namedparams())]
    ret = 0
    for p in params:
        shape = p.shape
        if len(shape) == 4:
            p = F.reshape(p, (shape[0], -1))
        elif len(shape) == 2:
            pass
        else:
            continue
        if p.shape[0] > p.shape[1]:
            p = F.transpose(p)
        xp = cuda.get_array_module(p.data)
        WTW = F.matmul(p, F.transpose(p))
        ret += F.sum(F.square(WTW - xp.eye(WTW.shape[0])))
    return ret
