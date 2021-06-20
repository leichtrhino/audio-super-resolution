#!/usr/env python

import unittest
import torch

import models

class ModelTest(unittest.TestCase):
    def test_multiscale_conv(self):
        batch_size = 2
        kernel_size = [3, 9, 27, 81]
        c = models.MultiscaleConv(
            in_channels=3,
            out_channels_per_conv=2,
            kernel_size=kernel_size
        )
        self.assertEqual(
            c(torch.rand(batch_size, 3, 50)).shape,
            torch.Size([batch_size, 2*len(kernel_size), 50])
        )

    def test_superpixel(self):
        batch_size = 2
        r_superpixel = 3
        s = models.Superpixel(r_superpixel=r_superpixel)

        # check shape
        self.assertEqual(
            s(torch.rand(batch_size, 3, 12)).shape,
            torch.Size([batch_size, 3*r_superpixel, 12 // r_superpixel])
        )
        # padded
        self.assertEqual(
            s(torch.rand(batch_size, 3, 10)).shape,
            torch.Size([batch_size, 3*r_superpixel, 12 // r_superpixel])
        )

        # check shuffle logic
        s = models.Superpixel(r_superpixel=2)
        self.assertTrue(
            torch.all(
                s(torch.arange(1, 9, dtype=float).reshape(1, 1, 8))
                == torch.Tensor([[[1, 3, 5, 7], [2, 4, 6, 8]]])
            )
        )

    def test_subpixel(self):
        batch_size = 2
        r_subpixel = 3
        s = models.Subpixel(r_subpixel=r_subpixel)

        # check shape
        self.assertEqual(
            s(torch.rand(batch_size, 9, 12)).shape,
            torch.Size([batch_size, 9 // r_subpixel, 12 * r_subpixel])
        )

        # check shuffle logic
        s = models.Subpixel(r_subpixel=2)
        self.assertTrue(
            torch.all(
                s(torch.Tensor([[[1, 3, 5, 7], [2, 4, 6, 8]]]))
                == torch.arange(1, 9, dtype=float).reshape(1, 1, 8)
            )
        )

    def test_downsampling_block(self):
        out_channels = 8
        r_superpixel = 2
        block = models.DownsamplingBlock(
            in_channels=2,
            out_channels=out_channels,
            kernel_size=[3, 9, 27, 81],
            r_superpixel=r_superpixel
        )

        self.assertEqual(
            block(torch.rand(8, 2, 50)).shape,
            torch.Size([8, out_channels, 50 // r_superpixel])
        )

    def test_upsampling_block(self):
        out_channels = 2
        r_subpixel = 2
        block = models.UpsamplingBlock(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=[3, 9, 27, 81],
            r_subpixel=r_subpixel,
            p_dropout=0.2
        )

        self.assertEqual(
            block(torch.rand(8, 16, 50)).shape,
            torch.Size([8, out_channels, 50 * r_subpixel])
        )

    def test_spline(self):
        s = models.Spline(upsample_rate=4)
        self.assertEqual(
            s(torch.rand(10)).shape,
            torch.Size([10*4])
        )
        self.assertEqual(
            s(torch.rand(8, 14)).shape,
            torch.Size([8, 14*4])
        )

    def test_generator(self):
        supersampling_rate = 2
        g = models.Generator(
            supersampling_rate=supersampling_rate,
            out_channel=[16, 32, 64],
            kernel_size=[3, 9, 27, 81],
            conv_in_channel=8,
            conv_kernel_size=27,
            superpixel_rate=2,
            dropout_p=0.2
        )
        self.assertEqual(
            g(torch.rand(8, 800)).shape,
            torch.Size([8, 800*supersampling_rate])
        )
        self.assertEqual(
            g(torch.rand(8, 802)).shape,
            torch.Size([8, 802*supersampling_rate])
        )

    def test_autoencoder(self):
        a = models.Autoencoder(
            out_channel=[16, 32, 64],
            kernel_size=[3, 9, 27, 81],
            conv_in_channel=8,
            conv_kernel_size=27,
            superpixel_rate=2,
            dropout_p=0.2
        )
        x_in = torch.rand(8, 802)
        encoder_out = a.encoder(x_in)
        decoder_out = a.decoder(encoder_out)
        autoencoder_out = a(x_in)
        self.assertEqual(
            encoder_out.shape,
            torch.Size([8, 64, 808 // (2**3)])
        )
        self.assertEqual(
            decoder_out.shape,
            torch.Size([8, 808]) # last 6 elements are padded
        )
        self.assertEqual(
            autoencoder_out.shape,
            x_in.shape # slice to the original shape
        )

    def test_discriminator(self):
        d = models.Discriminator(
            input_length=802,
            out_channel=[16, 32, 64],
            kernel_size=[3, 9, 27, 81],
            linear_out_features=1200,
            superpixel_rate=2,
            dropout_p=0.2,
        )
        x_in = torch.rand(8, 802)
        self.assertEqual(
            d(x_in).shape,
            torch.Size([8])
        )

class TrainTest(unittest.TestCase):

    def test_generator_loss(self):
        supersampling_rate = 2
        x_h = torch.rand(3, 1604)
        x_l = x_h[:, ::supersampling_rate]

        # gan components
        generator = models.Generator(
            supersampling_rate=supersampling_rate,
            out_channel=[8, 16],
            kernel_size=[3, 9],
            conv_in_channel=4,
            conv_kernel_size=9,
            superpixel_rate=2,
            dropout_p=0.2
        )
        discriminator = models.Discriminator(
            input_length=1604,
            out_channel=[8, 8, 16, 16],
            kernel_size=[3, 9],
            linear_out_features=10,
            superpixel_rate=2,
            dropout_p=0.2
        )
        autoencoder = models.Autoencoder(
            out_channel=[8, 16],
            kernel_size=[3, 9],
            conv_in_channel=4,
            conv_kernel_size=9,
            superpixel_rate=2,
            dropout_p=0.2
        )

        x_h_hat = generator(x_l)
        self.assertEqual(x_h.shape, x_h_hat.shape)

        # test L_L2
        self.assertTrue(
            torch.sum((x_h - x_h_hat) ** 2) / x_h.shape[-1] > 0
        )
        # test L_f
        feature_x_h = autoencoder.encoder(x_h)
        feature_x_h_hat = autoencoder.encoder(x_h_hat)
        self.assertEqual(feature_x_h.shape, feature_x_h_hat.shape)
        self.assertTrue(
            torch.sum((feature_x_h - feature_x_h_hat) ** 2
                      / (feature_x_h.shape[1] * feature_x_h.shape[2]) > 0)
        )
        # test L_adv
        p_x_h_hat = discriminator(x_h_hat)
        self.assertTrue(
            torch.sum(-torch.log(p_x_h_hat) > 0)
        )

    def test_discriminator_loss(self):
        supersampling_rate = 2
        x_h = torch.rand(3, 1604)
        x_l = x_h[:, ::supersampling_rate]

        # gan components
        generator = models.Generator(
            supersampling_rate=supersampling_rate,
            out_channel=[8, 16],
            kernel_size=[3, 9],
            conv_in_channel=4,
            conv_kernel_size=9,
            superpixel_rate=2,
            dropout_p=0.2
        )
        discriminator = models.Discriminator(
            input_length=1604,
            out_channel=[8, 8, 16, 16],
            kernel_size=[3, 9],
            linear_out_features=10,
            superpixel_rate=2,
            dropout_p=0.2
        )

        x_h_hat = generator(x_l)
        self.assertEqual(x_h.shape, x_h_hat.shape)

        # test L_D
        p_x_h = discriminator(x_h)
        p_x_h_hat = discriminator(x_h_hat)
        self.assertTrue(
            torch.sum(-torch.log(p_x_h_hat) -torch.log(1-p_x_h)) > 0
        )

if __name__ == '__main__':
    unittest.main()
