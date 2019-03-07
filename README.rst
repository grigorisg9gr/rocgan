======================
Robust Conditional GAN
======================

Chainer implementation of the ICLR paper "**Robust Conditional Generative Adversarial Networks**"
https://openreview.net/forum?id=Byg0DsCqYQ [1]_

Robust Conditional GANs aim at leveraging structure in the target space of the generator by augmenting it with a new, unsupervised pathway to learn the target structure. 

Testing/demo mode
=================

We provide a `jupyter notebook <https://github.com/grigorisg9gr/rocgan/blob/master/demo.ipynb>`_ that illustrates how to
call the code for testing.

If you want to use the pretrained models, please follow the instructions 
in the notebook. The demo testing images are  provided in the `demo/` folder. 

Train the network
=================

To train the network, e.g. with rocgan, you can execute the following command::

   python jobs/rocgan/train_mn.py '--config jobs/rocgan/iclr_5layer_rocgan_super.yml' 


Dataset preparation
===================

The default code for super-resolution (or any task that requires a pair of input/output
images) requires vertically concatenated images.
Please see the demo/ folder for some samples; the idea is to vertically concatenate
the images with the input (e.g. corrupted) image on top and the output image on
the bottom. 

Most of the other configurations are included in the `*.yml` that defines the 
modules/datasets to include in the training.

Misc
====

The results are improved over the original ICLR publication; those results and
pretrained files correspond to the journal version of the code (currently under
review).

Tested on a Linux machine with:
* chainer=4.0.0, chainercv=0.9.0,
* chainer=5.2.0, chainercv=0.12.0.

The code is highly influenced by [2]_.


Citing
======
If you use this code, please cite [1]_:

*BibTeX*:: 

  @inproceedings{
  chrysos2018rocgan,
  title={RoC-{GAN}: Robust Conditional {GAN}},
  author={Grigorios G. Chrysos and Jean Kossaifi and Stefanos Zafeiriou},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
  url={https://openreview.net/forum?id=Byg0DsCqYQ},
  }
  
References
==========

.. [1] Grigorios G. Chrysos, Jean Kossaifi and Stefanos Zafeiriou, **Robust Conditional Generative Adversarial Networks**, *International Conference on Learning Representations (ICLR)*, 2019.

.. [2] https://github.com/pfnet-research/sngan_projection
