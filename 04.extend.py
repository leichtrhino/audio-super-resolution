#!/usr/env python
import argparse
import resampy
import torch
import torchaudio

import datasets
import models
import model_settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args

def build_generator():
    return models.Generator(
        supersampling_rate=model_settings.supersampling_rate(),
        out_channel=model_settings.conv_out_channels(),
        kernel_size=model_settings.conv_kernel_size(),
        conv_in_channel=model_settings.out_conv_in_channel(),
        conv_kernel_size=model_settings.out_conv_kernel(),
        superpixel_rate=2,
        dropout_p=0.2
    )

def main():
    args = parse_args()

    # load audio file
    x, orig_sr = torchaudio.load(args.input)
    x_l = torch.Tensor(resampy.resample(
        x.numpy(),
        orig_sr,
        model_settings.sample_sr() // model_settings.supersampling_rate(),
        axis=-1
    )).to(args.device)

    # build (and load) a model
    generator = build_generator()
    checkpoint = torch.load(args.checkpoint)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(args.device)

    # extend
    generator.eval()
    with torch.no_grad():
        x_hat = generator(x_l).to('cpu')

    # save
    torchaudio.save(args.output, x_hat, sample_rate=model_settings.sample_sr())

if __name__ == '__main__':
    main()

