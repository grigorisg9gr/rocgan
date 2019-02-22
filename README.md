# Robust Conditional GAN
Chainer implementation of the paper Robust Conditional GAN
https://openreview.net/forum?id=Byg0DsCqYQ

# Testing/demo mode
We provide a jupyter notebook, i.e. demo.ipynb, that illustrates how to
call the code for testing.

If you want to use the pretrained models, please follow the instructions 
in the notebook. The demo testing images are already provided in the 
demo/ folder. 

# Train the network
To train the network, e.g. with rocgan, you can call it like this: 
``` 
python jobs/rocgan/train_mn.py '--config jobs/rocgan/iclr_5layer_rocgan_super.yml' 
```

# # Dataset preparation
The default code for super-resolution (or another task that a pair of input/output
images is expected) requires vertically concatenated images.
Please see the demo/ folder for some samples; the idea is to vertically concatenate
the images with the input (e.g. corrupted) image on top and the output image on
the bottom. 

Most of the other configurations are included in the *.yml that defines the 
modules/datasets to include in the training.

# Misc
The results are improved over the original ICLR publication; those results and
pretrained files correspond to the journal version of the code (currently under
review).

Tested on a Linux machine with:
* chainer=4.0.0, chainercv=0.9.0,
* chainer=5.2.0, chainercv=0.12.0.

The code is highly influenced from [1].


[1] https://github.com/pfnet-research/sngan_projection

