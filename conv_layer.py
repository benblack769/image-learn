import tensorflow as tf

def make_layer(name,shape):
    tot_size = reduce(mul, shape, 1)
    print(name,tot_size)
    rand_weight_vals = np.random.randn(tot_size).astype('float32')/(shape[-1]**(0.5**0.5))
    rand_weight = np.reshape(rand_weight_vals,shape)
    return tf.Variable(rand_weight,name=name)

def conv_cmp_layer(input_img, cmp_img, output_channels, cmp_hidden_size, dot_prod_size):

    return output_img, loss
