
from typing import List
from math import ceil
from numpy import linspace
from scipy.interpolate import splrep, splev

import torch

class Spline(torch.nn.Module):
    def __init__(self, upsample_rate : int) -> None:
        super(Spline, self).__init__()
        self.upsample_rate = upsample_rate

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # TODO: Tensor -> numpy -> Tensor transformation,
        # implement no-scipy style if possible later.
        assert len(x.shape) == 1 or len(x.shape) == 2

        w = x.shape[-1]
        t = lambda z: splev(
            linspace(0, w-1, w*self.upsample_rate),
            splrep(linspace(0, w-1, w), z),
            der=0
        )

        if len(x.shape) == 1:
            return torch.Tensor(t(x.cpu().numpy())).to(x.device)

        return torch.stack([torch.Tensor(t(z)) for z in x.cpu().numpy()])\
            .to(x.device)

class Subpixel(torch.nn.Module):
    def __init__(self, r_subpixel : int) -> None:
        super(Subpixel, self).__init__()
        self.r_subpixel = r_subpixel

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        n, c_in, l_in = x.shape

        assert c_in % self.r_subpixel == 0
        return x.reshape(n, c_in // self.r_subpixel, self.r_subpixel, l_in) \
                .transpose(2, 3) \
                .reshape(n, c_in // self.r_subpixel, l_in * self.r_subpixel)

class Superpixel(torch.nn.Module):
    def __init__(self, r_superpixel : int) -> None:
        super(Superpixel, self).__init__()
        self.r_superpixel = r_superpixel

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        r_pad = (self.r_superpixel - x.shape[-1] % self.r_superpixel) \
            % self.r_superpixel
        x = torch.nn.functional.pad(x, (0, r_pad), mode='reflect')
        n, c_in, l_in = x.shape

        assert l_in % self.r_superpixel == 0
        return x.reshape(n, c_in, l_in // self.r_superpixel, self.r_superpixel) \
                .transpose(2, 3) \
                .reshape(n, c_in * self.r_superpixel, l_in // self.r_superpixel)

class MultiscaleConv(torch.nn.Module):
    def __init__(
            self,
            in_channels : int,
            out_channels_per_conv : int,
            kernel_size : List[int]
    ) -> None:

        super(MultiscaleConv, self).__init__()
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels,
                out_channels_per_conv,
                k,
                padding=(k-1)//2
            )
            for k in kernel_size
        ])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.cat([c(x) for c in self.conv_layers], dim=1)

class DownsamplingBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels : int,
            out_channels : int,
            kernel_size : List[int],
            r_superpixel : int
    ) -> None:

        super(DownsamplingBlock, self).__init__()
        assert out_channels % (len(kernel_size) * r_superpixel) == 0
        out_channels_per_conv \
            = out_channels // (len(kernel_size) * r_superpixel)

        self.multiscale_conv = MultiscaleConv(
            in_channels, out_channels_per_conv, kernel_size)
        self.prelu = torch.nn.PReLU(init=0.2)
        self.superpixel = Superpixel(r_superpixel)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.superpixel(self.prelu(self.multiscale_conv(x)))

class UpsamplingBlock(torch.nn.Module):
    '''The upsampling blocks on the original paper include stacking from downsampling block.
    However, This impl. exclude it for (possible) flexibility
    '''

    def __init__(
            self,
            in_channels : int,
            out_channels : int,
            kernel_size : List[int],
            r_subpixel : int,
            p_dropout : float
    ) -> None:

        super(UpsamplingBlock, self).__init__()
        assert (out_channels * r_subpixel) % len(kernel_size) == 0
        out_channels_per_conv \
            = (out_channels * r_subpixel) // len(kernel_size)

        self.multiscale_conv = MultiscaleConv(
            in_channels, out_channels_per_conv, kernel_size)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.prelu = torch.nn.PReLU(init=0.2)
        self.subpixel = Subpixel(r_subpixel)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.subpixel(self.prelu(self.dropout(self.multiscale_conv(x))))

class Autoencoder(torch.nn.Module):

    class Encoder(torch.nn.Module):
        def __init__(
                self,
                out_channel : List[int],
                kernel_size : List[int],
                conv_in_channel : int,
                conv_kernel_size : int,
                superpixel_rate : int,
                dropout_p : float,
        ) -> None:

            super(Autoencoder.Encoder, self).__init__()
            self.downsampling_blocks = torch.nn.ModuleList([
                DownsamplingBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    r_superpixel=superpixel_rate
                )
                for in_c, out_c in zip([1]+out_channel, out_channel)
            ])

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            downsample_out = x.unsqueeze(1)
            for d in self.downsampling_blocks:
                downsample_out = d(downsample_out)
            return downsample_out


    class Decoder(torch.nn.Module):
        def __init__(
                self,
                out_channel : List[int],
                kernel_size : List[int],
                conv_in_channel : int,
                conv_kernel_size : int,
                superpixel_rate : int,
                dropout_p : float,
        ) -> None:

            super(Autoencoder.Decoder, self).__init__()

            self.upsampling_blocks = torch.nn.ModuleList([
                UpsamplingBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    r_subpixel=superpixel_rate,
                    p_dropout=dropout_p
                )
                for block_i, (in_c, out_c) in enumerate(zip(
                        out_channel[::-1],
                        out_channel[-2::-1]+[conv_in_channel]
                ))
            ])

            self.conv = torch.nn.Conv1d(
                conv_in_channel,
                1,
                conv_kernel_size,
                padding=(conv_kernel_size-1) // 2
            )

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            upsample_in = x
            for u in self.upsampling_blocks:
                upsample_in = u(upsample_in)
            return self.conv(upsample_in).squeeze(1)

    def __init__(
            self,
            out_channel : List[int],
            kernel_size : List[int],
            conv_in_channel : int,
            conv_kernel_size : int,
            superpixel_rate : int,
            dropout_p : float,
    ) -> None:

        super(Autoencoder, self).__init__()
        self.encoder = Autoencoder.Encoder(
            out_channel=out_channel,
            kernel_size=kernel_size,
            conv_in_channel=conv_in_channel,
            conv_kernel_size=conv_kernel_size,
            superpixel_rate=superpixel_rate,
            dropout_p=dropout_p,
        )
        self.decoder = Autoencoder.Decoder(
            out_channel=out_channel,
            kernel_size=kernel_size,
            conv_in_channel=conv_in_channel,
            conv_kernel_size=conv_kernel_size,
            superpixel_rate=superpixel_rate,
            dropout_p=dropout_p,
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-1]
        return self.decoder(self.encoder(x))[:, :orig_size]

class Generator(torch.nn.Module):
    def __init__(
            self,
            supersampling_rate : int,
            out_channel : List[int],
            kernel_size : List[int],
            conv_in_channel : int,
            conv_kernel_size : int,
            superpixel_rate : int,
            dropout_p : float,
    ) -> None:

        super(Generator, self).__init__()
        self.supersampling_rate = supersampling_rate
        self.out_channel = out_channel

        self.spline = Spline(supersampling_rate)
        self.downsampling_blocks = torch.nn.ModuleList([
            DownsamplingBlock(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel_size,
                r_superpixel=superpixel_rate
            )
            for in_c, out_c in zip([1]+out_channel, out_channel)
        ])

        # Each output of upsampling block is concat'ed with corresponding
        # downsampling block's output and feed to the next block.
        # So, double the input channels except the first block.
        self.upsampling_blocks = torch.nn.ModuleList([
             UpsamplingBlock(
                 in_channels=half_in_c * 2 if block_i > 0 else half_in_c,
                 out_channels=out_c,
                 kernel_size=kernel_size,
                 r_subpixel=superpixel_rate,
                 p_dropout=dropout_p
            )
            for block_i, (half_in_c, out_c) in enumerate(zip(
                    out_channel[::-1],
                    out_channel[-2::-1]+[conv_in_channel]
            ))
        ])

        self.conv = torch.nn.Conv1d(
            conv_in_channel,
            1,
            conv_kernel_size,
            padding=(conv_kernel_size-1) // 2
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.spline(x)
        orig_size = x.shape[-1]
        size_factor = self.supersampling_rate ** len(self.out_channel)
        r_pad = (size_factor - (orig_size % size_factor)) % size_factor
        x = torch.nn.functional.pad(
            x.unsqueeze(0), (0, r_pad), mode='reflect').squeeze(0)

        downsample_out = []
        for d in self.downsampling_blocks:
            downsample_out.append(d(
                downsample_out[-1] if downsample_out else x.unsqueeze(1)
            ))

        upsample_in = None
        for d_out, u in zip(downsample_out[::-1], self.upsampling_blocks):
            upsample_in = u(
                d_out if upsample_in is None else
                torch.cat([upsample_in, d_out], dim=1)
            )

        return (x + self.conv(upsample_in).squeeze(1))[:, :orig_size]

class Discriminator(torch.nn.Module):

    class DownsamplingBlock(torch.nn.Module):
        def __init__(
                self,
                in_channels : int,
                out_channels : int,
                kernel_size : List[int],
                r_superpixel : int,
                p_dropout : float
        ) -> None:

            super(Discriminator.DownsamplingBlock, self).__init__()
            assert out_channels % (len(kernel_size) * r_superpixel) == 0
            out_channels_per_conv \
                = out_channels // (len(kernel_size) * r_superpixel)

            self.multiscale_conv = MultiscaleConv(
                in_channels, out_channels_per_conv, kernel_size)
            self.batchnorm = torch.nn.BatchNorm1d(out_channels // r_superpixel)
            self.dropout = torch.nn.Dropout(p=p_dropout)
            self.leakyrelu = torch.nn.LeakyReLU()
            self.superpixel = Superpixel(r_superpixel)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            return self.superpixel(self.leakyrelu(
                self.dropout(self.batchnorm(self.multiscale_conv(x)))
            ))

    def __init__(
            self,
            input_length : int,
            out_channel : List[int],
            kernel_size : List[int],
            linear_out_features : int,
            superpixel_rate : int,
            dropout_p : float,
    ) -> None:
        super(Discriminator, self).__init__()
        assert out_channel[0] % len(kernel_size) == 0
        out_channels_per_conv = out_channel[0] // len(kernel_size)

        self.multiscale_conv = MultiscaleConv(
            1, out_channels_per_conv, kernel_size)
        self.leakyrelu_1 = torch.nn.LeakyReLU(0.2)

        self.downsampling_blocks = torch.nn.ModuleList([
            Discriminator.DownsamplingBlock(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel_size,
                r_superpixel=superpixel_rate,
                p_dropout=dropout_p
            )
            for in_c, out_c in zip(out_channel, out_channel[1:])
        ])
        # first multiscale conv. does not perform superpixel
        downsampling_length = ceil(
            input_length / superpixel_rate**(len(out_channel)-1))

        self.linear_1 = torch.nn.Linear(
            downsampling_length * out_channel[-1], linear_out_features)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.leakyrelu_2 = torch.nn.LeakyReLU(0.2)

        self.linear_2 = torch.nn.Linear(linear_out_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        downsample_out = x.unsqueeze(1)
        downsample_out = self.leakyrelu_1(self.multiscale_conv(
            downsample_out))

        for d in self.downsampling_blocks:
            downsample_out = d(downsample_out)

        linear_out = self.leakyrelu_2(self.dropout(
            self.linear_1(downsample_out.flatten(start_dim=1))))

        return self.sigmoid(self.linear_2(linear_out)).squeeze(1)

