import pdb
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.layers


def leaky_relu(x, name=None):
    return tf.maximum(x, 0.2*x, name=name)


def GeneratorCNN(z, num_filters, filter_size, channels_out, repeat_num,
        data_format, reuse):
    """Maps (batch_size, z_dim) to (batch_size, 4, 4, num_filters) to
     (batch_size, scale_size, scale_size, 3).
    """
    act = leaky_relu
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([4, 4, num_filters]))
        x = layers.dense(z, num_output)
        x = tf.reshape(x, [-1, 4, 4, num_filters])
        
        for idx in range(repeat_num):
            num_filters /= 2
            x = layers.conv2d_transpose(x, num_filters, filter_size, 2,
                padding='same', use_bias=False, activation=None)
            x = layers.batch_normalization(x)
            x = act(x)

        out = layers.conv2d_transpose(x, channels_out, filter_size, 2,
            padding='same', use_bias=False, activation=tf.nn.tanh)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def AutoencoderCNN(x, input_channel, z_num, repeat_num, num_filters,
        filter_size, data_format, reuse, to_decode=None):
    """Maps (batch_size, scale_size, scale_size, 3) to 
      (batch_size, 4, 4, num_filters) to (batch_size, z_dim), and reverse.
    """
    verbose = False 
    if verbose:
        print(x)
    log2_num_filter = int(np.log2(num_filters))
    act = leaky_relu
    channels_out = x.shape.as_list()[-1]
    with tf.variable_scope("ae_enc", reuse=reuse) as vs_enc:
        # Encoder
        for idx in range(repeat_num + 1):
            channel_num = 2 ** (log2_num_filter - repeat_num + idx)
            x = layers.conv2d(x, channel_num, filter_size, 2,
                padding='same', use_bias=False, activation=None)
            x = layers.batch_normalization(x)
            x = act(x, name='act{}'.format(idx))
            if verbose:
                print(x)

        final_conv_flat_dim = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, final_conv_flat_dim])
        if verbose:
            print(x)
        z = x = layers.dense(x, z_num)
        if verbose:
            print(x)
        if to_decode is not None:
            x = to_decode

    with tf.variable_scope("ae_dec", reuse=reuse) as vs_dec:
        # Decoder
        x = layers.dense(x, final_conv_flat_dim)
        if verbose:
            print(x)
        x = tf.reshape(x, [-1, 4, 4, num_filters])
        if verbose:
            print(x)
        
        for idx in range(repeat_num):
            num_filters /= 2
            x = layers.conv2d_transpose(x, num_filters, filter_size, 2,
                padding='same', use_bias=False, activation=None)
            x = layers.batch_normalization(x)
            x = act(x)
            if verbose:
                print(x)

        out = layers.conv2d_transpose(x, channels_out, filter_size, 2,
            padding='same', use_bias=False, activation=tf.nn.tanh)
        if verbose:
            print(out)
            pdb.set_trace()

    variables_enc = tf.contrib.framework.get_variables(vs_enc)
    variables_dec = tf.contrib.framework.get_variables(vs_dec)
    return out, z, variables_enc, variables_dec


def MMD(data, gen, t_mean, t_cov_inv, sigma=1):
    '''Using encodings of data and generated samples, compute MMD.

    Args:
      data: Tensor of encoded data samples.
      gen: Tensor of encoded generated samples.
      t_mean: Tensor, mean of batch of encoded target samples.
      t_cov_inv: Tensor, covariance of batch of encoded target samples.
      sigma: Scalar lengthscale of MMD kernel.

    Returns:
      mmd: Scalar, metric of discrepancy between the two samples.
    '''
    xe = data
    ge = gen
    data_num = tf.shape(xe)[0]
    gen_num = tf.shape(ge)[0]
    v = tf.concat([xe, ge], 0)
    VVT = tf.matmul(v, tf.transpose(v))
    sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
    sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
    exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
    K = tf.exp(-0.5 * (1 / sigma) * exp_object)
    K_yy = K[data_num:, data_num:]
    K_xy = K[:data_num, data_num:]
    K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) - 
                  tf.matrix_band_part(K_yy, 0, 0))
    num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)

    def prob_of_keeping(xi):
        xt_ = xi - tf.transpose(t_mean)
        x_ = tf.transpose(xt_)
        pr = 1. - 0.5 * tf.exp(-10. * tf.matmul(tf.matmul(xt_, t_cov_inv), x_))
        return pr

    keeping_probs = tf.reshape(tf.map_fn(prob_of_keeping, xe), [-1, 1])
    keeping_probs_tiled = tf.tile(keeping_probs, [1, gen_num])
    p1_weights_xy = 1. / keeping_probs_tiled
    p1_weights_xy_normed = p1_weights_xy / tf.reduce_sum(p1_weights_xy)
    Kw_xy = K[:data_num, data_num:] * p1_weights_xy_normed
    mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
           2 * tf.reduce_sum(Kw_xy))


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


###############################################################################
def predict_weights_from_enc(x, dropout_pr, reuse):
    """mnist_enc_NN builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, z_dim), where z_dim is the
      number of encoding dimension.
      dropout_pr: tf.float32 indicating the keeping rate for dropout.
    Returns:
      y_logits: Tensor of shape (N_examples, 2), with values equal to the logits
        of classifying the digit into zero/nonzero.
      y_probs: Tensor of shape (N_examples, 2), with values
        equal to the probabilities of classifying the digit into zero/nonzero.
    """
    act = leaky_relu
    z_dim = x.get_shape().as_list()[1]
    with tf.variable_scope('mnist_classifier', reuse=reuse) as vs:
        x = slim.fully_connected(x, 1024, activation_fn=act, scope='fc1')
        x = slim.dropout(x, dropout_pr, scope='drop1')
        x = slim.fully_connected(x, 1024, activation_fn=act, scope='fc2')
        x = slim.dropout(x, dropout_pr, scope='drop2')
        x = slim.fully_connected(x, 32, activation_fn=act, scope='fc3')
        x = slim.dropout(x, dropout_pr, scope='drop3')
        y = slim.fully_connected(x, 1, activation_fn=None, scope='fc4')
        #y_probs = tf.nn.softmax(y_logits)

        '''
        fc_dim = 1024
        W_fc1 = weight_variable([z_dim, fc_dim])
        b_fc1 = bias_variable([fc_dim])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout_pr)

        W_fc2 = weight_variable([fc_dim, fc_dim])
        b_fc2 = bias_variable([fc_dim])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, dropout_pr)

        W_fc3 = weight_variable([fc_dim, 2])
        b_fc3 = bias_variable([2])

        y_logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
        y_probs = tf.nn.softmax(y_logits)
        '''

    variables = tf.contrib.framework.get_variables(vs)
    return y, variables
