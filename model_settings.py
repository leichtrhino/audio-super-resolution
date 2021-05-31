
import math

def supersampling_rate():
    return 2

def conv_kernel_size():
    return [3, 9, 27, 81]

def conv_out_channels():
    return [64, 128, 256, 512]

def out_conv_in_channel():
    return 32

def out_conv_kernel():
    return 27

def sample_sr():
    return 44100

def sample_duration():
    return sample_segment_length() / sample_sr()

def sample_segment_length():
    orig_segments = math.ceil(sample_sr() * 0.05)
    segment_factor = supersampling_rate()**len(conv_out_channels())
    return math.ceil(orig_segments / segment_factor) * segment_factor

def out_linear_features():
    return 1024
