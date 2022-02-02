
import math

def supersampling_rate():
    return 2

def conv_kernel_size():
    return [3, 9, 27, 81]

def conv_out_channels(layer: int):
    base_channels = [128, 256, 512, 512, 512, 512, 512, 512][:layer]
    return [c // 2 for c in base_channels]

def out_conv_in_channel():
    return 32

def out_conv_kernel():
    return 27

def sample_sr():
    return 16000

def sample_duration():
    return sample_segment_length() / sample_sr()

def sample_segment_length():
    orig_segments = math.ceil(sample_sr() * 4.0)
    segment_factor = supersampling_rate()**len(conv_out_channels(layer=8))
    return math.ceil(orig_segments / segment_factor) * segment_factor

def out_linear_features():
    return 1024
