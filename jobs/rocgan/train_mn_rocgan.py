import os, sys
from socket import gethostname
from time import strftime
import argparse
import chainer
from chainer import training
from chainer.training import extensions, extension
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
import chainermn
import multiprocessing

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
sys.path.append(base)

from evaluations.extensions import sample_generate, sample_generate_conditional, sample_generate_light, calc_inception
from evaluations.extensions_cgan import validation_trainer
import yaml
import source.yaml_utils as yaml_utils
from source.miscs.model_moiving_average import ModelMovingAverage
from source.misc_train_utils import (create_result_dir, load_models_cgan, printtime,
                                     ensure_config_paths, plot_losses_log)


def make_optimizer(model, comm, alpha=0.001, beta1=0.9, beta2=0.999, chmn=False, add_decay=False):
    # # 12/2018: problem in minoas, probably related with openmpi.
    if chmn:
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2), comm)
    else:
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    if add_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    #optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(0.1), 'hook_clip')
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--n_devices', type=int)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--communicator', type=str,
                        default='hierarchical', help='Type of communicator')
    parser.add_argument('--results_dir', type=str, default='results_rocgan')
    parser.add_argument('--inception_model_path', type=str,
                        default='/home/user/inception/inception.model')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--enc_snapshot', type=str, default=None, help='path to the encoder (reg pathway) snapshot')
    parser.add_argument('--enc_ae_snapshot', type=str, default=None, help='path to the encoder (ae pathway) snapshot')
    parser.add_argument('--dec_snapshot', type=str, default=None, help='path to the decoder snapshot')
    parser.add_argument('--dis_snapshot', type=str, default=None, help='path to the discriminator snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--multiprocessing', action='store_true', default=False)
    parser.add_argument('--validation', type=int, default=1)
    parser.add_argument('--valid_fn', type=str, default='files_valid_4k.txt', 
                        help='filename of the validation file')
    parser.add_argument('--label', type=str, default='synth')
    parser.add_argument('--stats_fid', type=str, default='', help='path for FID stats')
    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # # ensure that the paths of the config are correct.
    config = ensure_config_paths(config)
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()
    # # get the pc name, e.g. for chainerui.
    pcname = gethostname()
    imperialpc = 'doc.ic.ac.uk' in pcname or pcname in ['ladybug', 'odysseus']
    print('Init on pc: {}.'.format(pcname))
    if comm.rank == 0:
        print('==========================================')
        print('Using {} communicator'.format(args.communicator))
        print('==========================================')
    enc, dec, dis, enc_ae = load_models_cgan(config, rocgan=True)
    if chainer.cuda.available:
        enc.to_gpu()
        dec.to_gpu()
        dis.to_gpu()
    else:
        print('No GPU found!!!\n')
    mma1 = ModelMovingAverage(0.999, enc)
    mma2 = ModelMovingAverage(0.999, dec)
    models = {'enc': enc, 'dec': dec, 'dis': dis, 'enc_ae': enc_ae}
    if args.enc_snapshot is not None:
        print('Loading encoder (reg pathway): {}.'.format(args.enc_snapshot))
        chainer.serializers.load_npz(args.enc_snapshot, enc)
    if args.enc_ae_snapshot is not None:
        print('Loading encoder (ae pathway): {}.'.format(args.enc_ae_snapshot))
        chainer.serializers.load_npz(args.enc_ae_snapshot, enc_ae)
    if args.dec_snapshot is not None:
        print('Loading decoder: {}.'.format(args.dec_snapshot))
        chainer.serializers.load_npz(args.dec_snapshot, dec)
    if args.dis_snapshot is not None:
        print('Loading discriminator: {}.'.format(args.dis_snapshot))
        chainer.serializers.load_npz(args.dis_snapshot, dis)
    # # convenience function for optimizer:
    func_opt = lambda net: make_optimizer(net, comm, chmn=args.multiprocessing,
                                          alpha=config.adam['alpha'], beta1=config.adam['beta1'], 
                                          beta2=config.adam['beta2'])
    # Optimizer
    opt_enc = func_opt(enc)
    opt_dec = func_opt(dec)
    opt_dis = func_opt(dis)
    opt_enc_ae = func_opt(enc_ae)
    opts = {'opt_enc': opt_enc, 'opt_dec': opt_dec, 'opt_dis': opt_dis, 'opt_enc_ae': opt_enc_ae}
    # Dataset
    if comm.rank == 0:
        dataset = yaml_utils.load_dataset(config)
        if args.validation:
            # # add the validation db if we do perform validation.
            db_valid = yaml_utils.load_dataset(config, validation=True, valid_path=args.valid_fn)
    else:
        _ = yaml_utils.load_dataset(config)  # Dummy, for adding path to the dataset module
        dataset = None
        if args.validation:
            _ = yaml_utils.load_dataset(config, validation=True, valid_path=args.valid_fn)
            db_valid = None
    dataset = chainermn.scatter_dataset(dataset, comm)
    if args.validation:
        db_valid = chainermn.scatter_dataset(db_valid, comm)
    # Iterator
    multiprocessing.set_start_method('forkserver')
    if args.multiprocessing:
        # # In minoas this might fail with the forkserver.py error.
        iterator = chainer.iterators.MultiprocessIterator(dataset, config.batchsize,
                                                          n_processes=args.loaderjob)
        if args.validation:
            iter_val = chainer.iterators.MultiprocessIterator(db_valid, config.batchsize,
                                                              n_processes=args.loaderjob,
                                                              shuffle=False, repeat=False)
    else:
        iterator = chainer.iterators.SerialIterator(dataset, config.batchsize)
        if args.validation:
            iter_val = chainer.iterators.SerialIterator(db_valid, config.batchsize,
                                                        shuffle=False, repeat=False)
    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': iterator,
        'optimizer': opts,
        'device': device,
        'mma1': mma1,
        'mma2': mma2,
    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    if imperialpc:
        mainf = '{}_{}'.format(strftime('%Y_%m_%d__%H_%M_%S'), args.label)
        out = os.path.join(args.results_dir, mainf, '')
    elif not args.test:
        out = args.results_dir
    else:
        out = 'results/test'
    if comm.rank == 0:
        create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    # # abbreviations below: inc -> incpetion, gadv -> grad_adv, lgen -> loss gener, 
    # # {m, sd, b}[var] -> {mean, std, best position} [var], 
    report_keys = ['loss_dis', 'lgen_adv', 'dis_real', 'dis_fake', 'loss_l1', 'lrec_l1',
                   'mssim', 'sdssim', 'mmae', 'loss_projl', 'llat', 'FID']

    if comm.rank == 0:
        # Set up logging
        for m in models.values():
            trainer.extend(extensions.snapshot_object(
                m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))
#         trainer.extend(extensions.snapshot_object(
#             mma.avg_model, mma.avg_model.__class__.__name__ + '_avgmodel_{.updater.iteration}.npz'),
#             trigger=(config.snapshot_interval, 'iteration'))
        trainer.extend(extensions.LogReport(trigger=(config.display_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
        if args.validation:
            # # add the appropriate extension for validating the model.
            models_mma = {'enc': mma1.avg_model, 'dec': mma2.avg_model, 'dis': dis}
            trainer.extend(validation_trainer(models_mma, iter_val, n=len(db_valid), export_best=True, 
                                              pout=out, p_inc=args.inception_model_path, eval_fid=True,
                                              sfile=args.stats_fid),
                           trigger=(config.evaluation_interval, 'iteration'),
                           priority=extension.PRIORITY_WRITER)

        trainer.extend(extensions.ProgressBar(update_interval=config.display_interval))
        if imperialpc:
            # [ChainerUI] Observe learning rate
            trainer.extend(extensions.observe_lr(optimizer_name='opt_dis'))
            # [ChainerUI] enable to send commands from ChainerUI
            trainer.extend(CommandsExtension())
            # [ChainerUI] save 'args' to show experimental conditions
            save_args(args, out)

    # # convenience function for linearshift in optimizer:
    func_opt_shift = lambda optim1: extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                                           (config.iteration_decay_start, 
                                                            config.iteration), optim1)
    # # define the actual extensions (for optimizer shift).
    trainer.extend(func_opt_shift(opt_enc))
    trainer.extend(func_opt_shift(opt_dec))
    trainer.extend(func_opt_shift(opt_dis))
    trainer.extend(func_opt_shift(opt_enc_ae))

    if args.resume:
        print('Resume Trainer')
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    printtime('start training')
    trainer.run()
    plot_losses_log(out, savefig=True)


if __name__ == '__main__':
    main()
