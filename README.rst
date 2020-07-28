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

The yml file describes the modules/dataset to train on. The hyper-parameters are included
in the yml, no need to hardcode them in the files.


Dataset preparation
===================

The code is tuned for super-resolution (or any task that requires a pair of input/output
images). 

In each iteration, a single file is loaded from disk; this file is the vertical concatenation of
the input and the output images. 
Please see the demo/ folder for some samples; the idea is to vertically concatenate
the images with the input (e.g. corrupted) image on top and the output image on
the bottom. 
All images are resized to 64x64 for training/testing.


Browsing the folders
====================
The folder structure is the following:

*    ``gen_models``: The folder for the generator models.

*    ``dis_models``: The folder for the discriminator models; do not modify the optional arguments, unless you know what you are doing.

*    ``updater``: The folder that contains the core code for the updater, i.e. the chunk of code that runs in every iteration to update the modules.

*    ``jobs``: It contains a) the yml with the hyper-parameter setting, b) the main command to load the models and run the training (train_mn*.py).

*    ``source``: It contains auxiliary code, you should probably not modify any of that code.

*    ``evaluations``: It contains the code for validation (either during training or offline).

Misc
====

The results are improved over the original ICLR publication; those results and
pretrained files correspond to the IJCV version [4]_.

Tested on a Linux machine with:

* chainer=4.0.0, chainercv=0.9.0,

* chainer=5.2.0, chainercv=0.12.0.


The code is highly influenced by [2]_.

Apart from Chainer, the code depends on Pyaml [3]_. 


Citing
======
If you use this code, please cite [1]_:

*BibTeX*:: 

  @inproceedings{
  chrysos2018rocgan,
  title={Robust Conditional Generative Adversarial Networks},
  author={Grigorios G. Chrysos and Jean Kossaifi and Stefanos Zafeiriou},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019},
  url={https://openreview.net/forum?id=Byg0DsCqYQ},
  }

or:

  @article{
  chrysos2020rocgan,
  title={RoCGAN: Robust Conditional GAN},
  author={Grigorios G. Chrysos and Jean Kossaifi and Stefanos Zafeiriou},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2020},
  publisher={Springer}
  }


  
References
==========

.. [1] Grigorios G. Chrysos, Jean Kossaifi and Stefanos Zafeiriou, **Robust Conditional Generative Adversarial Networks**, *International Conference on Learning Representations (ICLR)*, 2019.

.. [2] https://github.com/pfnet-research/sngan_projection

.. [3] https://pypi.org/project/pyaml/

.. [4] Grigorios G. Chrysos, Jean Kossaifi and Stefanos Zafeiriou, **RoCGAN: Robust Conditional GAN**, *International Journal of Computer Vision*, 2020.

