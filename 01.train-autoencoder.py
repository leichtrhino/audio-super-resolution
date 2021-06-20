#!/usr/env python
import sys
import argparse
import torch

import datasets
import models
import model_settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--validation-dir', type=str)
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

def build_model():
    return models.Autoencoder(
        out_channel=model_settings.conv_out_channels(layer=4),
        kernel_size=model_settings.conv_kernel_size(),
        conv_in_channel=model_settings.out_conv_in_channel(),
        conv_kernel_size=model_settings.out_conv_kernel(),
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
        shuffle=True
    )

    if args.validation_dir is not None:
        validation_dataset = datasets.Folder(
            args.validation_dir,
            model_settings.sample_sr(),
            model_settings.sample_duration()
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    # build (and load) a model
    autoencoder = build_model()
    if args.checkpoint:
        autoencoder.load_state_dict(torch.load(args.checkpoint))
    autoencoder.to(args.device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), 1e-4)

    for epoch in range(1, args.epoch+1):
        sum_loss = 0
        total_batch = 0
        ave_loss = 0
        last_output_len = 0

        # train
        autoencoder.train()
        for step, batch in enumerate(train_loader, 1):

            optimizer.zero_grad()
            for sample_i in range(0, batch.shape[0], args.compute_batch_size):
                # obtain batch and infer then get loss
                sample_i_end = min(
                    sample_i + args.compute_batch_size, batch.shape[0])
                x = batch[sample_i:sample_i_end].to(args.device)
                x_hat = autoencoder(x)
                loss = loss_function(x, x_hat)

                # obtain training informationn
                sum_loss += loss.item() * x.shape[0]
                total_batch += x.shape[0]
                ave_loss = sum_loss / total_batch

                # perform a backward pass
                loss *= (sample_i_end - sample_i) / batch.shape[0]
                loss.backward()

            optimizer.step()

            # print learning statistics
            print_step = step
            curr_output = f'\repoch {epoch} step {print_step} loss={ave_loss}'
            sys.stdout.write('\r' + ' ' * last_output_len)
            sys.stdout.write(curr_output)
            sys.stdout.flush()
            last_output_len = len(curr_output)

        if args.validation_dir is None:
            sys.stdout.write('\n')
            continue

        # validation
        autoencoder.eval()
        with torch.no_grad():
            sum_val_loss = 0
            total_batch = 0

            for x in validation_loader:
                # obtain batch, infer, get loss
                x = x.to(args.device)
                x_hat = autoencoder(x)
                loss = loss_function(x, x_hat)

                # update training statistics
                sum_val_loss += loss.item() * x.shape[0]
                total_batch += x.shape[0]

        ave_val_loss = sum_val_loss / total_batch
        sys.stdout.write('\r' + ' ' * last_output_len)
        sys.stdout.write(f'\repoch {epoch} loss={ave_loss} val={ave_val_loss}\n')

    autoencoder.to('cpu')
    torch.save(autoencoder.state_dict(), args.output)

if __name__ == '__main__':
    main()

