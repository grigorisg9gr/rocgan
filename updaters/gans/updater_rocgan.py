import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
import source.functions.losses as losses
from source.miscs.random_samples import sample_categorical


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        assert not self.conditional, 'Conditional modules deleted here!'
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        self.mma1 = kwargs.pop('mma1') if 'mma1' in kwargs else None
        self.mma2 = kwargs.pop('mma2') if 'mma2' in kwargs else None
        self.update_gener = kwargs.pop('update_gener') if 'update_gener' in kwargs else True
        self.upd_less_ae = kwargs.pop('upd_less_ae') if 'upd_less_ae' in kwargs else False
        self.upd_ae_freq = kwargs.pop('upd_ae_freq') if 'upd_ae_freq' in kwargs else 3
        self.upd_less_st = kwargs.pop('upd_less_st') if 'upd_less_st' in kwargs else 1000
        # # in cgan we have additional losses in generator, added as a list.
        self.add_loss_gen = kwargs.pop('add_loss_gen') if 'add_loss_gen' in kwargs else []
        self.add_loss_dis = kwargs.pop('add_loss_dis') if 'add_loss_dis' in kwargs else []
        # # additional kwargs for different losses.
        self.l1_weight = kwargs.pop('l1_weight') if 'l1_weight' in kwargs else 1
        self.recl1_weight = kwargs.pop('recl1_weight') if 'recl1_weight' in kwargs else 1
        self.latl_weight = kwargs.pop('latl_weight') if 'latl_weight' in kwargs else 1
        self.projl_weight = kwargs.pop('projl_weight') if 'projl_weight' in kwargs else 1
        self.decovl_weight = kwargs.pop('decovl_weight') if 'decovl_weight' in kwargs else 1
        if self.loss_type == 'softplus':
            self.loss_gen = losses.loss_dcgan_gen
            self.loss_dis = losses.loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = losses.loss_hinge_gen
            self.loss_dis = losses.loss_hinge_dis
        elif self.loss_type == 'revKL':
            self.loss_gen = losses.loss_revKL_gen
            self.loss_dis = losses.loss_revKL_dis
        elif self.loss_type == 'wgan':
            self.loss_gen = losses.loss_wgan_gen
            self.loss_dis = losses.loss_wgan_dis

        super(Updater, self).__init__(*args, **kwargs)

    def _generate_samples(self, x_real, ae_pathway=False):
        # # decide the encoder based on the pathway.
        enc = self.models['enc'] if not ae_pathway else self.models['enc_ae']
        dec = self.models['dec']
        if hasattr(dec, 'skip_enc') and dec.skip_enc is not None:
            lat, enc_outs = enc(x_real, save_res=True)
            x_fake = dec(lat, skips=enc_outs)
        else:
            # # in this case, no lateral connections, the 
            # # encoder's output is provided to the decoder.
            lat = enc(x_real)
            x_fake = dec(lat)
        # # in rocgan, the third return argument is devoted to 
        # # latent representation (required for latent/decov loss).
        return x_fake, None, lat

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x, y = [], []
        for j in range(batchsize):
            # # The iterator only accepts one image,
            # # so use the function here to assign first
            # # 3 channels to corrupted and last 3 to gt.
            x.append(np.asarray(batch[j][0][:3]).astype('f'))
            if batch[j][0].shape[0] > 3:
                y.append(np.asarray(batch[j][0][3:]).astype('f'))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y)) if len(y) > 0 else x_real
        return x_real, y_real

    def total_loss_gen(self, dis_fake, x_fake, y_real, y_recon=None, 
                       lat_reg=None, lat_ae=None):
        """
        Function to compute the total loss of the generator.
        """
        # # adversarial loss for generator.
        lgen = self.loss_gen(dis_fake=dis_fake)
        chainer.reporter.report({'lgen_adv': lgen.array})
        loss = lgen + 0
        # # loop over the additional loss types.
        for lt in self.add_loss_gen:
            if lt == 'l1':
                # # L1 loss.
                loss_l1 = F.mean_absolute_error(y_real, x_fake)
                chainer.reporter.report({'loss_l1': loss_l1.array})
                loss += self.l1_weight * loss_l1
            elif lt == 'rec_l1':
                # # L1 loss (ae pathway).
                loss_recl1 = F.mean_absolute_error(y_real, y_recon)
                chainer.reporter.report({'lrec_l1': loss_recl1.array})
                loss += self.recl1_weight * loss_recl1
            elif lt == 'latl':
                # # latent loss.
                loss_lat = F.mean_absolute_error(lat_reg, lat_ae)
                chainer.reporter.report({'llat': loss_lat.array})
                loss += self.latl_weight * loss_lat
            elif lt == 'decovl':
                # # decov loss.
                loss_decov = losses.decov_loss(lat_ae)
                chainer.reporter.report({'ldecov': loss_decov.array})
                loss += self.decovl_weight * loss_decov
            elif lt == 'latl2':
                # # latent loss.
                loss_lat = F.mean_squared_error(lat_reg, lat_ae)
                chainer.reporter.report({'llat': loss_lat.array})
                loss += self.latl_weight * loss_lat
            else:
                m1 = 'Not recognized loss type ({}).'
                raise RuntimeError(m1.format(lt))
        return loss

    def total_loss_dis(self, dis_fake, dis_real, freal, ffake):
        """
        Function to compute the total loss of the discriminator.
        """
        # # adversarial loss for discriminator.
        ldis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
        chainer.reporter.report({'loss_dis': ldis.array})
        loss = ldis + 0
        # # loop over the additional loss types.
        for lt in self.add_loss_dis:
            if lt == 'projl':
                # # projection loss in the discriminator features.
                loss_proj = F.mean_absolute_error(freal, ffake)
                chainer.reporter.report({'loss_projl': loss_proj.array})
                loss += self.projl_weight * loss_proj
            else:
                m1 = 'Not recognized loss type ({}).'
                raise RuntimeError(m1.format(lt))
        return loss

    def update_core(self):
        enc = self.models['enc']
        dec = self.models['dec']
        dis = self.models['dis']
        enc_ae = self.models['enc_ae']
        enc_optimizer = self.get_optimizer('opt_enc')
        dec_optimizer = self.get_optimizer('opt_dec')
        dis_optimizer = self.get_optimizer('opt_dis')
        enc_ae_optimizer = self.get_optimizer('opt_enc_ae')
        xp = dec.xp
        # # initialize the discriminator features in both cases as None.
        freal, ffake = None, None
        for i in range(self.n_dis):
            # # in cgan, x_real is input image (condition) and y_real is the target image.
            x_real, y_real = self.get_batch(xp)
            if 'projl' in self.add_loss_dis:
                # # in this case, we want the representation.
                dis_real, freal = dis(y_real, return_feature=True)
            else:
                dis_real = dis(y_real)
            # # in cgan, x_fake is the output of the generator.
            x_fake, y_fake, _ = self._generate_samples(x_real)
            if 'projl' in self.add_loss_dis:
                # # in this case, we want the representation.
                dis_fake, ffake = dis(x_fake, y=y_fake, return_feature=True)
            else:
                dis_fake = dis(x_fake, y=y_fake)
            x_fake.unchain_backward()

            fake_arr, real_arr = dis_fake.array, dis_real.array
            chainer.reporter.report({'dis_fake': fake_arr.mean()})
            chainer.reporter.report({'dis_real': real_arr.mean()})
            chainer.reporter.report({'fake_max': fake_arr.max()})
            chainer.reporter.report({'real_min': real_arr.min()})

            loss_dis = self.total_loss_dis(dis_fake, dis_real, freal, ffake)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            loss_dis.unchain_backward()
            del loss_dis

            if i == self.n_dis - 1 and self.update_gener:
                # # Run a second time for updating the generator.
                x_fake, y_fake, lat_reg = self._generate_samples(x_real)
                # # since we do not update discriminator, we do not need features here.
                dis_fake = dis(x_fake, y=y_fake)
                # # call the generation for the ae path.
                y_recon, _, lat_ae = self._generate_samples(y_real, ae_pathway=True)
                if self.upd_less_ae and self.iteration % self.upd_ae_freq != 0 and self.iteration > self.upd_less_st:
                    lat_ae.unchain_backward()
                    y_recon.unchain_backward()
                loss_gen = self.total_loss_gen(dis_fake, x_fake, y_real, y_recon, 
                                               lat_reg, lat_ae)
                enc.cleargrads()
                enc_ae.cleargrads()
                dec.cleargrads()
                loss_gen.backward()
                dec_optimizer.update()
                enc_optimizer.update()
                enc_ae_optimizer.update()
                loss_gen.unchain_backward()

                del loss_gen, lat_reg, lat_ae

        self.mma2.update(dec) if self.mma2 is not None else None
        self.mma1.update(enc) if self.mma1 is not None else None

