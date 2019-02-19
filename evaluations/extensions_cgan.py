import os
import sys
import math
from os.path import join, isdir, dirname, isfile

import numpy as np
from PIL import Image
import scipy.linalg

import chainer
import chainer.cuda
from chainer import Variable, serializers, cuda
import chainer.functions as F
from chainer.training import extensions

sys.path.append(dirname(__file__))
sys.path.append('../')
from source.functions.msssim import RGBSSIM, _calc_ssim
# # define as globals since the trainer does not have a memory.
best_ssim, best_mae, best_pos_ssim = -1, 1000, 0


def get_batch(iterator, xp):
    batch = iterator.next()
    batchsize = len(batch)
    x, y = [], []
    for j in range(batchsize):
        # # The iterator only accepts one image,
        # # so use the function here to assign first
        # # 3 channels to corrupted and last 3 to gt.
        x.append(np.asarray(batch[j][0][:3]).astype('f'))
        if batch[j][0].shape[0] > 3:
            y.append(np.asarray(batch[j][0][3:]).astype('f'))
        elif len(batch[j]) > 2:
            # # case where the iterator returns multiple images, assumed
            # # that it returns image, label, image.
            y.append(np.asarray(batch[j][2]).astype('f'))
    x_real = Variable(xp.asarray(x))
    y_real = Variable(xp.asarray(y)) if len(y) > 0 else x_real
    return x_real, y_real


def save_image(x, rows, cols, dst, name):
    _, _, h, w = x.shape
    x = x.reshape((rows, cols, 3, h, w))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * h, cols * w, 3))
    preview_dir = join(dst, 'preview', '')
    preview_path = preview_dir + '{}.png'.format(name)
    if not isdir(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)


def deprocess_image_var(x):
    """ Given a var image, copy it to cpu and convert it to uint8 (from [-1, 1])."""
    x = chainer.cuda.to_cpu(x.data)
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    return x


def gen_images_cgan(enc, dec, iterator, n=4000, func=None):
    """
    Gets the inputs and targets of the generator (encoder-decoder)
    network; it applies the network and returns the reconstructions.
    Optionally applies a func after the decoder.
    """
    # # corrupted, gt and generated images lists.
    ims_cor, ims_gt, ims_gen = [], [], []
    xp = dec.xp
    for i in range(0, n, iterator.batch_size):
        x_real, y_real = get_batch(iterator, xp)
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', func is not None):
            if hasattr(dec, 'skip_enc') and dec.skip_enc is not None:
                # # we need to save outputs of encoder (skip case).
                lat, enc_outs = enc(x_real, save_res=True)
                x = dec(lat, skips=enc_outs)
            else:
                lat = enc(x_real)
                x = dec(lat)
            if func is not None:
                # # apply the function provided as an argument.
                x_real, y_real, x = func(x_real, y_real, x)
        # # append each image to the respective list.
        ims_gen.append(deprocess_image_var(x))
        ims_cor.append(deprocess_image_var(x_real))
        ims_gt.append(deprocess_image_var(y_real))
    ims_cor = np.concatenate(ims_cor, 0)
    ims_gen = np.concatenate(ims_gen, 0)
    ims_gt = np.concatenate(ims_gt, 0)
    return ims_gen, ims_cor, ims_gt


def compute_FID(ims, path_inception, stat_file, batchsize=100):
    """ Given the images ims, it computes the Frechet Inception Distance. """
    # # relative imports.
    from extensions import FID, get_mean_cov, load_inception_model
    # # load the inception model.
    model = load_inception_model(path_inception)
    # # load the stat file, i.e. the mean and cov 
    # # for the clean images.
    stat = np.load(stat_file)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, ims, batch_size=batchsize)
    fid = FID(stat['mean'], stat['cov'], mean, cov)
    chainer.reporter.report({'FID': fid})
    return fid


def evaluate_cond_gener(enc, dec, iterator, n=4000, seed=None, eval_ssim=True, 
                        eval_fid=False, path_inception=None, stat_file=None):
    """
    It accepts a model and an iterator and evaluates the generated
    images (conditional generator) against the gt with various metrics.
    ARGS:
        seed: (int, optional) Set the seed; used for reducing randomness, e.g. 
            in dataset transformations.
        path_inception: (str, optional) Path of the inception pretrained 
            model; provide for evaluating FID. 
        stat_file: (str, optional) Path of the stat file with the mean/cov for
            the FID. This is db-specific (distribution-specific) metric.
    """
    ssim = RGBSSIM()
    if seed is not None:
        # # optionally set the seed.
        np.random.seed(seed)
    # # get the images (generated, corrupted and gt).
    ims_gen, ims_cor, ims_gt = gen_images_cgan(enc, dec, iterator, n=n)
    # # loop over the sampled images and compare the metrics.
    ssims, maes, fid = [], [], -1
    for ige, ic, igt in zip(ims_gen, ims_cor, ims_gt):
        if eval_ssim:
            ssims.append(_calc_ssim(ssim, ige, igt))
        maes.append(np.mean(np.abs(ige - igt)))
    if eval_fid and isfile(path_inception):
        fid = compute_FID(ims_gen, path_inception, stat_file)
    # # convert the outputs to numpy's.
    ssims = np.array(ssims, dtype=np.float32)
    maes = np.array(maes, dtype=np.float32)
    # # format in a dictionary (easier to extend in the future).
    metrics = {'ssim': ssims, 'mae': maes, 'fid': fid}
    return metrics, [ims_gen, ims_cor, ims_gt]


def _mean_std(metric, dec=3):
    return np.round(np.mean(metric), dec), np.round(np.std(metric), dec)

def _reduce_concat_lists(list_of_lists, n_red):
    """ Given a list of lists it reduces the length of each 
        to n_red; it concatenates the list. """
    lfinal = None
    for cnt, l in enumerate(list_of_lists):
        lnew = l[:n_red]
        if cnt == 0:
            lfinal = lnew.copy()
        else:
            lfinal = np.concatenate((lfinal, lnew), axis=-2)
    return lfinal
    

def validation_trainer(models, iterator, n=4000, seed=0, eval_ssim=True, 
                       export_best=False, iter_start=1000, n_exp=10, 
                       pout=None, eval_fid=True, p_inc=None, sfile=None):
    """
    Generates a chainer extension for evaluating the reconstruction in cgan.
    ARGS:
        n_exp: (int, optional) The number of images to be exported for best
            validation score.
        p_inc: (str, optional) Path of the inception pretrained 
            model; provide for evaluating FID. 
        sfile: (str, optional) Path of the stat file with the mean/cov for
            the FID. This is db-specific (distribution-specific) metric.
    """
    @chainer.training.make_extension()
    def evaluation(trainer=None):
        enc, dec = models['enc'], models['dec']
        iterator.reset()
        # # evaluate the metrics.
        metrics, imsl = evaluate_cond_gener(enc, dec, iterator, n=n, 
                                            seed=seed, eval_ssim=eval_ssim,
                                            eval_fid=eval_fid, path_inception=p_inc,
                                            stat_file=sfile)
        # # compute mean and std of metrics.
        ssim_m, ssim_std = _mean_std(metrics['ssim'])
        mae_m, mae_std = _mean_std(metrics['mae'], dec=1)
        if trainer is None:
            # # early stop if no trainer is provided.
            return
        chainer.reporter.report({
        'mssim': ssim_m, 'sdssim': ssim_std, 
        'mmae': mae_m, 'sdmae': mae_std
        })
        # # ensure the iteration is at least iter_start. 
        iter1 = trainer.updater.iteration
        if export_best and iter1 > iter_start:
            # # in this case, we want to check if we have the
            # # best validation score and export the model.
            global best_ssim, best_mae, best_pos_ssim
            if ssim_m > best_ssim:
                # # export the models/sample images (best validation score).
                print('Debugging, validation (exporting), iter={}!'.format(iter1))
                for m in models.values():
                    ext = extensions.snapshot_object(m, m.__class__.__name__ + '_best.npz')
                    ext(trainer)
                # # update the best ssim.
                best_ssim, best_pos_ssim = ssim_m, iter1
                chainer.reporter.report({'bssim': best_pos_ssim})
                # # adapt the number of images to be exported.
                n_ims_exp = min(n_exp, imsl[0].shape[0])
                # # assume that the data are saved in imsl in this format; 
                # # hardcoded based on the above.
                # # reduce the lists length and concatenate the images.
                ims_conc = _reduce_concat_lists(imsl, n_ims_exp)
                save_image(ims_conc, 1, n_ims_exp, pout, 'reconst')
    return evaluation


def sample_generate_light(enc, gen, iterator, dst, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        iterator.reset()
        n_images = rows * cols
        x = gen_images(enc, gen, iterator, n_images)
        save_image(x, rows, cols, dst, 'image_latest')

    return make_image


def sample_generate_reconstruction(enc, dec, iterator, dst, rows=5, cols=5, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        iterator.reset()
        np.random.seed(seed)
        n_images = rows * cols
        x_gen, x_cor, x_gt = gen_images_cgan(enc, dec, iterator, n_images)
        save_image(x_gen, rows, cols, dst, 'image_rec_gen{:0>8}'.format(trainer.updater.iteration))
        save_image(x_cor, rows, cols, dst, 'image_rec_cor')
        save_image(x_gt, rows, cols, dst, 'image_rec_gt')

    return make_image



