
import math

def supersampling_rate():
    return 2

def conv_kernel_size():
    #return [3, 9, 27, 81]
    return [3, 9]

def conv_out_channels():
    #return [64, 128, 256, 512]
    return [8, 16, 32, 32]

def out_conv_in_channel():
    return 32

def out_conv_kernel():
    return 27

def sample_sr():
    return 44100

def sample_duration():
    return 3.0

def sample_segment_length():
    orig_segments = math.ceil(sample_sr() * sample_duration())
    segment_factor = supersampling_rate()**len(conv_out_channels())
    return math.ceil(orig_segments / segment_factor) * segment_factor

def out_linear_features():
    return math.ceil(math.sqrt(max(
        conv_out_channels()[0],
        sample_segment_length()
        // (supersampling_rate()**len(conv_out_channels()))
    )))
