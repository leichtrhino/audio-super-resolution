#!/usr/env python
import math
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
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args

def build_generator():
    return models.Generator(
        supersampling_rate=model_settings.supersampling_rate(),
        out_channel=model_settings.conv_out_channels(layer=8),
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
    ))
    orig_length = x_l.shape[-1]
    sample_num = math.ceil(
        x_l.shape[-1] / model_settings.sample_segment_length())

    x_l = torch.cat((
        x_l,
        torch.zeros(
            *x_l.shape[:-1],
            model_settings.sample_segment_length() * sample_num - x_l.shape[-1]
        )
    ), dim=-1).reshape(
        *x_l.shape[:-1], sample_num, model_settings.sample_segment_length()
    )

    # build (and load) a model
    generator = build_generator()
    checkpoint = torch.load(args.checkpoint)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    generator.to(args.device)

    # extend
    out_tensors = []
    for channel_i, channel in enumerate(x_l):
        for batch_i in range(0, channel.shape[0], args.batch_size):
            batch_end_i = min(batch_i + args.batch_size, channel.shape[0])
            batch = channel[batch_i:batch_end_i]
            with torch.no_grad():
                x_hat = generator(batch.to(args.device)).to('cpu').flatten()

            if len(out_tensors) == channel_i:
                out_tensors.append(x_hat)
            else:
                out_tensors[channel_i] = torch.cat((
                    out_tensors[channel_i], x_hat))

    x_hat = torch.stack(out_tensors)[:, :orig_length]

    # save
    torchaudio.save(args.output, x_hat, sample_rate=model_settings.sample_sr())

if __name__ == '__main__':
    main()

