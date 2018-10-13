import numpy as np

def make_gather_conv(conv_size,input_size):
    res = []
    conv_size_sqr = conv_size*conv_size
    for x in range(conv_size):
        for y in range(conv_size):
            for i in range(input_size):
                for j in range(input_size*conv_size_sqr):
                    res.append(int(j // input_size == x * conv_size + y))
    res_np = np.asarray(res,dtype=np.float32)
    res_reshaped = np.reshape(res_np,(conv_size,conv_size,input_size,conv_size*conv_size*input_size))
    return res_reshaped
