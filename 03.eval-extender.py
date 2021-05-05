#!/usr/env python
import argparse
import torch

import datasets
import models
import model_settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-dir', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    if args.gpu:
        args.device = 'gpu'
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

def snr(x : torch.Tensor, x_ref : torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(torch.mean(torch.clamp(
        torch.sum(x_ref**2, dim=-1)
        / torch.sum(torch.clamp((x-x_ref)**2, min=1e-16)),
        min=1e-16
    )))

def main():
    args = parse_args()

    # build dataset
    validation_dataset = datasets.Folder(
        args.validation_dir,
        model_settings.sample_sr(),
        duration=None
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False
    )

    # build (and load) a model
    generator = build_generator()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['generator'])
    generator.to(args.device)

    # evaluation
    print('sample_i,snr')
    generator.eval()
    with torch.no_grad():
        for sample_i, batch in enumerate(validation_loader):
            # obtain batch, infer
            x_h = batch.to(args.device)
            x_l = x_h[:, ::model_settings.supersampling_rate()]
            x_h_hat = generator(x_l)

            # align input and output
            x_length = min(x_h.shape[-1], x_h_hat.shape[-1])
            x_h = x_h[:, :x_length]
            x_h_hat = x_h_hat[:, :x_length]

            print(f'{sample_i},{snr(x_h, x_h_hat)}')

if __name__ == '__main__':
    main()

