import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


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
