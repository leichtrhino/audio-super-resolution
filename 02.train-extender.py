#!/usr/env python
import sys
import argparse
import resampy
import torch

import datasets
import models
import model_settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--validation-dir', type=str)
    parser.add_argument('--autoencoder', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--compute-batch-size', type=int, default=None)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('output', type=str)

    args = parser.parse_args()
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    if args.compute_batch_size is None:
        args.compute_batch_size = args.batch_size
    return args

def build_autoencoder():
    return models.Autoencoder(
        out_channel=model_settings.conv_out_channels(layer=4),
        kernel_size=model_settings.conv_kernel_size(),
        conv_in_channel=model_settings.out_conv_in_channel(),
        conv_kernel_size=model_settings.out_conv_kernel(),
        superpixel_rate=2,
        dropout_p=0.2
    )

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

def build_discriminator():
    return models.Discriminator(
        input_length=model_settings.sample_segment_length(),
        out_channel=model_settings.conv_out_channels(layer=8),
        kernel_size=model_settings.conv_kernel_size(),
        linear_out_features=model_settings.out_linear_features(),
        superpixel_rate=2,
        dropout_p=0.2
    )

def main():
    args = parse_args()

    # build dataset
    train_dataset = datasets.Folder(
        args.train_dir,
        model_settings.sample_sr(),
        model_settings.sample_duration()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    if args.validation_dir is not None:
        validation_dataset = datasets.Folder(
            args.validation_dir,
            model_settings.sample_sr(),
            model_settings.sample_duration()
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.compute_batch_size,
            shuffle=False,
            num_workers=4,
        )

    # build (and load) a model
    autoencoder = build_autoencoder()
    autoencoder.load_state_dict(torch.load(args.autoencoder)['model'])
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    autoencoder.to(args.device)

    generator = build_generator()
    discriminator = build_discriminator()
    optimizer_g = torch.optim.Adam(generator.parameters(), 1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), 1e-4)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['generator'])
        if 'generator_optimizer':
            optimizer_g.load_state_dict(checkpoint['generator_optimizer'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        if 'discriminator_optimizer':
            optimizer_d.load_state_dict(checkpoint['discriminator_optimizer'])

    generator.to(args.device)
    for state in optimizer_g.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to(args.device)
    discriminator.to(args.device)
    for state in optimizer_d.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to(args.device)

    for epoch in range(1, args.epoch+1):
        sum_loss_g = 0
        sum_loss_d = 0
        total_batch = 0
        last_output_len = 0
        # train
        discriminator.train()
        generator.train()

        for step, batch in enumerate(train_loader, 1):
            # obtain batch
            total_batch += batch.shape[0]

            # train discriminator
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False
            optimizer_d.zero_grad()

            # obtain batch and infer then get loss
            for sample_i in range(0, batch.shape[0], args.compute_batch_size):
                sample_i_end = min(
                    sample_i + args.compute_batch_size, batch.shape[0])
                x_h = batch[sample_i:sample_i_end]
                x_l = torch.Tensor(resampy.resample(
                    x_h.numpy(),
                    model_settings.sample_sr(),
                    model_settings.sample_sr()
                    // model_settings.supersampling_rate(),
                    axis=-1
                )).to(args.device)
                x_h = x_h.to(args.device)

                x_h_hat = generator(x_l)
                p_x_h = discriminator(x_h)
                p_x_h_hat = discriminator(x_h_hat)

                loss_d = torch.sum(
                    -torch.log(p_x_h.clamp(min=1e-32))
                    -torch.log((1-p_x_h_hat).clamp(min=1e-32))
                ) / x_h.shape[0]
                sum_loss_d += loss_d.item() * x_h.shape[0]

                loss_d *= (sample_i_end - sample_i) / batch.shape[0]
                loss_d.backward()

            optimizer_d.step()

            # train generator
            for p in generator.parameters():
                p.requires_grad = True
            for p in discriminator.parameters():
                p.requires_grad = False
            optimizer_g.zero_grad()

            # obtain batch and infer then get loss
            for sample_i in range(0, batch.shape[0], args.compute_batch_size):
                sample_i_end = min(
                    sample_i + args.compute_batch_size, batch.shape[0])
                x_h = batch[sample_i:sample_i_end]
                #x_l = x_h[:, ::model_settings.supersampling_rate()]
                x_l = torch.Tensor(resampy.resample(
                    x_h.numpy(),
                    model_settings.sample_sr(),
                    model_settings.sample_sr()
                    // model_settings.supersampling_rate(),
                    axis=-1
                )).to(args.device)
                x_h = x_h.to(args.device)

                x_h_hat = generator(x_l)[:, :x_h.shape[-1]]
                p_x_h_hat = discriminator(x_h_hat)
                f_x_h = autoencoder.encoder(x_h)
                f_x_h_hat = autoencoder.encoder(x_h_hat)

                loss_g = (
                    torch.sum((x_h - x_h_hat) ** 2) / x_h.shape[-1] \
                    + 1.0 * torch.sum((f_x_h - f_x_h_hat) ** 2) \
                    / (f_x_h.shape[1] * f_x_h.shape[2]) \
                    + 0.001 * torch.sum(-torch.log(p_x_h_hat.clamp(min=1e-32)))
                ) / x_h.shape[0]
                sum_loss_g += loss_g.item() * x_h.shape[0]

                loss_g *= (sample_i_end - sample_i) / batch.shape[0]
                loss_g.backward()

            optimizer_g.step()

            # print learning statistics
            print_step = step
            curr_output = (f'\repoch {epoch} step {print_step} '
                           f'loss_g={sum_loss_g / total_batch} '
                           f'loss_d={sum_loss_d / total_batch}')
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        if args.validation_dir is None:
            sys.stdout.write('\n')
            continue

        # validation
        generator.eval()
        discriminator.eval()
        sum_val_loss_g = 0
        sum_val_loss_d = 0
        total_val_batch = 0

        for batch in validation_loader:
            # obtain batch
            x_h = batch
            x_l = torch.Tensor(resampy.resample(
                x_h.numpy(),
                model_settings.sample_sr(),
                model_settings.sample_sr()
                // model_settings.supersampling_rate(),
                axis=-1
            )).to(args.device)
            x_h = x_h.to(args.device)

            with torch.no_grad():
                x_h_hat = generator(x_l)[:, :x_h.shape[-1]]
                p_x_h = discriminator(x_h)
                p_x_h_hat = discriminator(x_h_hat)
                f_x_h = autoencoder.encoder(x_h)
                f_x_h_hat = autoencoder.encoder(x_h_hat)

            total_val_batch += batch.shape[0]

            # eval discriminator
            loss_d = torch.sum(
                -torch.log(p_x_h.clamp(min=1e-32))
                -torch.log((1-p_x_h_hat).clamp(min=1e-32))
            )
            sum_val_loss_d += loss_d.item()

            # eval generator
            loss_g = (
                torch.sum((x_h - x_h_hat) ** 2) / x_h.shape[-1] \
                + 1.0 * torch.sum((f_x_h - f_x_h_hat) ** 2) \
                / (f_x_h.shape[1] * f_x_h.shape[2]) \
                + 0.001 * torch.sum(-torch.log(p_x_h_hat.clamp(min=1e-32)))
            )
            sum_val_loss_g += loss_g.item()

        # print learning statistics
        sys.stdout.write('\r' + ' ' * last_output_len)
        sys.stdout.write(f'\repoch {epoch} '
                         f'loss_g={sum_loss_g / total_batch} '
                         f'loss_d={sum_loss_d / total_batch} '
                         f'val_g={sum_val_loss_g / total_val_batch} '
                         f'val_d={sum_val_loss_d / total_val_batch}\n')

    generator.to('cpu')
    discriminator.to('cpu')
    for state in optimizer_g.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to('cpu')
    for state in optimizer_d.state.values():
        for k, v in state.items():
            if type(v) == torch.Tensor:
                state[k] = v.to('cpu')

    torch.save({
        'generator': generator.state_dict(),
        'generator_optimizer': optimizer_g.state_dict(),
        'discriminator': discriminator.state_dict(),
        'discriminator_optimizer': optimizer_d.state_dict(),
    }, args.output)

if __name__ == '__main__':
    main()

