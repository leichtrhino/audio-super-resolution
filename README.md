# audio-super-resolution

Modified and unofficial implementation of the audio super-res network proposed by [Kim and Sathe].

### Summary

This repo is modified and unofficial implementation of the audio super-res network proposed by Kim and Sathe [1].
The original discriminator network has FC, dropout, and leakyReLU layer on top of 8 downsampling blocks.
To reduce the network size, I replaced that part with a global maxpooling layer.

### Dependencies

- python 3.7.9
- pytorch 1.5.0
- torchaudio 0.5.0
- resampy 0.2.2

### Usage

#### Setting hyperparameters

Edit `model_settings.py` to match desired experimental setup.

- `supersampling_rate()`
- `conv_kernel_size()`: a list of filter sizes for multiscale conv. layers
- `conv_out_channels()`: a list of output feature sizes for the corresponding downsampling/upsampling blocks
- `out_conv_in_channel()`: feature size of the conv. layer on the top of the autoencoders and generator networks.
- `out_conv_kernel()`: kernel size of the conv. layer on the top of the autoencoders and generator networks.
- the first constant in `sample_segment_length()`: duration of training/validation sample size in seconds.

#### Training

Run `01.train-autoencoder.py` to train the autoencoder then run `02.train-extender.py` to train the GAN.

#### Prediction and evaluation

Simple evaluation and prediction program are also provided.
Run `03.eval-extender.py` for evaluation and `04.extend.py` for prediction (make super resolution of any .wav file)

[Kim and Sathe]: https://arxiv.org/abs/1903.09027
