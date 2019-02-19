import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


def decov_loss(tensor, xp=None, axis=1):
    """
    Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'.
    This version implements the loss in the variable format.

    ARGS:
        axis: (int, optional) If the tensor is 4-dim, it is
            reshaped into 2-dim; axis is the first dimension.
    """
    if xp is None:
        # # get xp module if not provided.
        xp = chainer.cuda.get_array_module(tensor.data)
    if tensor.ndim == 4:
        # # reshape to a 2D matrix.
        matr = F.reshape(tensor, (tensor.shape[axis], -1))
    elif tensor.ndim == 2:
        matr = tensor
    # # subtract the mean.
    centered = F.bias(matr, -F.mean(matr))
    # # compute the covariance.
    cov = F.matmul(centered, F.transpose(centered))
    # # compute the frombenius norm.
    frob_norm = F.sum(F.square(cov))
    # # get the norm of diagonal elements.
    # # in chainer 5.x this should work.
#     corr_diag_sqr = F.sum(F.square(F.diagonal(cov1)))
    corr_diag_sqr = F.sum(F.square(cov * xp.eye(cov.shape[0], dtype=cov.dtype)))
    loss = 0.5 * (frob_norm - corr_diag_sqr)
    return loss


def decov_loss_matrix(tensor, xp=None, axis=1):
    """
    Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'.
    This version implements the matrix case, i.e. Variables are converted
    into matrices/tensors.

    ARGS:
        axis: (int, optional) If the tensor is 4-dim, it is
            reshaped into 2-dim; axis is the first dimension.
    """
    if type(tensor, Variable):
        tensor = tensor.data
    if xp is None:
        # # get xp module if not provided.
        xp = chainer.cuda.get_array_module(tensor)
    if tensor.ndim == 4:
        # # reshape to a 2D matrix.
        matr = tensor.reshape((tensor.shape[axis], -1))
    elif tensor.ndim == 2:
        matr = tensor.copy()
    # # subtract the mean.
    mean1 = matr.mean()
    matr -= mean1
    # # compute the covariance.
    cov = xp.matmul(matr, matr.T)
    # # compute the frombenius norm.
    frob_norm = xp.sum(xp.square(cov))
    # # get the norm of diagonal elements.
    corr_diag_sqr = xp.sum(xp.square(xp.diag(cov)))
    loss = 0.5 * (frob_norm - corr_diag_sqr)
    return loss


def loss_revKL_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_revKL_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


# WGAN Loss
def loss_wgan_dis(dis_fake, dis_real):
    loss = - F.mean(dis_real) + F.mean(dis_fake)
    return loss


def loss_wgan_gen(dis_fake):
    loss = - F.mean(dis_fake)
    return loss


def gradient_penalty(dis_output, x):
    grad_x, = chainer.grad([dis_output], [x], enable_double_backprop=True)
    norm_grad_x = F.mean(F.sum(grad_x * grad_x, axis=(1, 2, 3)))
    return norm_grad_x


def gradient_penalty_wgangp(dis_output, x, lipnorm):
    grad_x, = chainer.grad([dis_output], [x], enable_double_backprop=True)
    norm_grad_x = F.sqrt(F.sum(grad_x * grad_x, axis=(1, 2, 3)))
    xp = norm_grad_x.xp
    return F.mean_squared_error(norm_grad_x, lipnorm * xp.ones_like(norm_grad_x.array))
