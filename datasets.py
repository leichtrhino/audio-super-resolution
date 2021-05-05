
import os
import math
import bisect
import pathlib
import random

import torch
import torchaudio
import resampy

class Folder(torch.utils.data.Dataset):
    def __init__(self, root_dir, sr, duration=None, transform=None):
        self.sr = sr
        self.duration = duration
        self.transform = transform
        self.offsets = [0]
        self.rates = []

        self.paths = sorted(list(pathlib.Path(root_dir).glob('**/*.wav')))

        for p in self.paths:
            si, _ = torchaudio.info(str(p))
            self.rates.append(si.rate)
            if self.duration is None:
                self.offsets.append(self.offsets[-1] + 1)
                continue
            if torchaudio.get_audio_backend() in ('sox', 'sox_io'):
                n_frames = si.length // si.channels
            elif torchaudio.get_audio_backend() == 'soundfile':
                n_frames = si.length
            n_segments = math.ceil(n_frames / si.rate / self.duration)
            self.offsets.append(self.offsets[-1] + n_segments)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        audio_idx = bisect.bisect(self.offsets, idx) - 1
        offset_idx = idx - self.offsets[audio_idx]
        if self.duration is None:
            offset = 0
            num_frames = 0
        else:
            offset = offset_idx * int(self.duration * self.rates[audio_idx])
            num_frames = int(self.rates[audio_idx] * self.duration)
        x, _ = torchaudio.load(
            str(self.paths[audio_idx]), offset=offset, num_frames=num_frames
        )
        x = x.mean(dim=0)
        if x.shape[-1] * self.sr / self.rates[audio_idx] < 1:
            x = torch.zeros((
                *x.shape[:-1], math.ceil(self.rates[audio_idx] / self.sr)
            ))
        x = torch.Tensor(resampy.resample(
            x.numpy(), self.rates[audio_idx], self.sr, axis=-1
        ))
        if self.duration is not None:
            out_length = int(self.sr * self.duration)
            if x.shape[-1] > out_length:
                x = x[:self.out_length]
            if x.shape[-1] < out_length:
                x = torch.cat((x, torch.zeros(out_length-x.shape[-1])))
        if self.transform is not None:
            x = self.transform(x)
        return x

