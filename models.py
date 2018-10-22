from collections import namedtuple
import pdb
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.layers


def activation_choice(x, name=None):
    #return tf.maximum(x, 0.2*x, name=name)  # Leaky RELU.
    #return tf.minimum(tf.nn.elu(x, name=name), 2)
    return tf.nn.elu(x, name=name)
    #return tf.nn.relu6(x, name=name)


def conv2d(x_orig, num_filters, filter_size, use_bias=False,
        batch_resid=False, extra_dense=False, resize_scale=None):
    # Do batch residual.
    if batch_resid:
        x = layers.conv2d(x_orig, num_filters, filter_size, strides=1,
            padding='same', use_bias=use_bias, activation=None)
        x_ = layers.batch_normalization(x)
        x_ = activation_choice(x_)

        x_ = layers.conv2d(x_, num_filters, filter_size, strides=1,
            padding='same', use_bias=use_bias, activation=None)
        x_ = layers.batch_normalization(x_)
        res = activation_choice(x_)

        out = tf.add(x, res, name='conv_batch_resid')

        if extra_dense:
            batch_size, current_dim, _, current_num_filters = out.shape.as_list()
            flat_dim = np.prod([current_dim, current_dim, current_num_filters])
            x_flat = tf.reshape(out, [-1, flat_dim])
            x_dense = dense_batch_resid(x_flat, use_bias=use_bias, activation=activation_choice)
            out = tf.reshape(x_dense, [-1, current_dim, current_dim, current_num_filters]) 
    else:
        x = layers.conv2d(x_orig, num_filters, filter_size, strides=1,
            padding='same', use_bias=use_bias, activation=None)
        x = layers.batch_normalization(x)
        out = activation_choice(x)

    # Resize.
    if resize_scale:
        current_dim = out.shape.as_list()[1]
        out = resize(out, int(resize_scale * current_dim))
        # Alternatively, use strided convolution instead of this fn.

    return out


def dense_batch_resid(x, use_bias=False, activation=None):
    x_dim = x.shape.as_list()[1]
    x_ = layers.batch_normalization(x)
    x_ = activation_choice(x_)
    x_ = layers.dense(x_, x_dim, use_bias=use_bias, activation=None)

    x_ = layers.batch_normalization(x_)
    x_ = activation_choice(x_)
    x_ = layers.dense(x_, x_dim, use_bias=use_bias, activation=None)

    r = tf.add(x_, x, name='dense_batch_resid')
    return r


def resize(x, new_size):
    new_dims = tf.convert_to_tensor([new_size, new_size])
    x = tf.image.resize_nearest_neighbor(x, new_dims)
    return x


def GeneratorCNN(z, base_size, num_filters, filter_size, channels_out, repeat_num,
        data_format, reuse, use_bias=False, verbose=False):
    """Maps (batch_size, z_dim) to (batch_size, base_size, base_size, num_filters) to
       (batch_size, scale_size, scale_size, 3).
       Output is on [-1,1].
    """
    if verbose:
        print('\n\nGENERATOR ARCHITECTURE\n')
        print(z)
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([base_size, base_size, num_filters]))
        x = layers.dense(z, num_output)
        if verbose: print(x)
        x = tf.reshape(x, [-1, base_size, base_size, num_filters])
        if verbose: print(x)
        
        current_dim = base_size
        current_num_filters = num_filters
        for idx in range(repeat_num):
            current_num_filters /= 2
            x = conv2d(x, current_num_filters, filter_size, use_bias=use_bias,
                batch_resid=False, extra_dense=False, resize_scale=2)
            current_dim *= 2

            if verbose: print(x)

        x = layers.conv2d(x, channels_out, filter_size, 1,
            padding='same', use_bias=use_bias, activation=None)
        x = tf.nn.tanh(x)
        out = resize(x, 2 * current_dim)

        if verbose: print(out)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def AutoencoderCNN(x, base_size, scale_size, input_channel, z_num, repeat_num,
        num_filters, filter_size, data_format, reuse, to_decode=None,
        use_bias=False, verbose=False):
    """Maps (batch_size, scale_size, scale_size, 3) to 
      (batch_size, base_size, base_size, num_filters) to (batch_size, z_dim),
      and reverse.
    """
    verbose = True 
    if verbose:
        print('\n\nAUTOENCODER ARCHITECTURE\n')
        print(x)

    channels_out = x.shape.as_list()[-1]
    current_num_filters = 2 ** (int(np.log2(num_filters)) - repeat_num)

    with tf.variable_scope("ae_enc", reuse=reuse) as vs_enc:
        # Encoder
        current_dim = scale_size
        for idx in range(repeat_num + 1):
            x = conv2d(x, current_num_filters, filter_size, use_bias=use_bias,
                batch_resid=False, extra_dense=False, resize_scale=0.5)
            current_dim /= 2
            current_num_filters *= 2

            if verbose: print(x)

        final_conv_flat_dim = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, final_conv_flat_dim])

        if verbose: print(x)

        # Operations on hidden layer.
        z = x = layers.dense(x, z_num)
        #hidden = z = x = dense_batch_resid(hidden, use_bias=use_bias, activation=activation_choice)
        #z = layers.batch_normalization(x) 
        #hidden = z = x = tf.nn.tanh(hidden) 
        #hidden = z = x = hidden 
        z_to_decode = tf.nn.dropout(z, 1) 

        if verbose:
            print(z_to_decode)

    with tf.variable_scope("ae_dec", reuse=reuse) as vs_dec:
        # Decoder
        x = layers.dense(z_to_decode, final_conv_flat_dim)
        if verbose: print(x)
        x = tf.reshape(x, [-1, base_size, base_size, num_filters])
        if verbose: print(x)
        
        current_dim = base_size
        current_num_filters = num_filters 
        for idx in range(repeat_num):
            current_num_filters /= 2
            x = conv2d(x, current_num_filters, filter_size, use_bias=use_bias,
                batch_resid=False, extra_dense=False, resize_scale=2)
            current_dim *= 2

            if verbose: print(x)

        x = layers.conv2d(x, channels_out, filter_size, 1,
            padding='same', use_bias=use_bias, activation=None)
        out = resize(x, 2 * current_dim)

        if verbose: print(out)

    variables_enc = tf.contrib.framework.get_variables(vs_enc)
    variables_dec = tf.contrib.framework.get_variables(vs_dec)
    return out, z, variables_enc, variables_dec


def DiscriminatorCNN(x, base_size, scale_size, input_channel, repeat_num,
        num_filters, filter_size, data_format, reuse, to_decode=None,
        use_bias=False, verbose=False):
    """Maps (batch_size, scale_size, scale_size, 3) to 
      (batch_size, base_size, base_size, num_filters) to (batch_size, z_dim),
      and reverse.
    """
    verbose = True 
    if verbose:
        print('\n\nDISCRIMINATOR ARCHITECTURE\n')
        print(x)

    channels_out = x.shape.as_list()[-1]
    current_num_filters = 2 ** (int(np.log2(num_filters)) - repeat_num)

    with tf.variable_scope("discrim", reuse=reuse) as vs_discrim:
        # Discrim, like encoder. 
        current_dim = scale_size
        for idx in range(repeat_num + 1):
            x = conv2d(x, current_num_filters, filter_size, use_bias=use_bias,
                batch_resid=False, extra_dense=False, resize_scale=0.5)
            current_dim /= 2
            current_num_filters *= 2

            if verbose: print(x)

        final_conv_flat_dim = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, final_conv_flat_dim])
        if verbose: print(x)
        logits = layers.dense(x, 1)

        if verbose: print(logits)

        variables = tf.contrib.framework.get_variables(vs_discrim)
        return logits, variables 


# TODO: ResNET
def resnet_conv2d(x_orig, num_filters, filter_size, training=True,
                  do_maxpool=False, verbose=False):
    simple = True
    if simple:
        if verbose: print(x_orig)
        net = layers.batch_normalization(x_orig, training=training)
        net = tf.nn.relu(net)
        net = layers.conv2d(
            net, 
            filters=num_filters,
            kernel_size=filter_size,
            padding='same',
            activation=None)
        if verbose: print(net)
        net = layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = layers.conv2d(
            net, 
            filters=num_filters,
            kernel_size=filter_size,
            padding='same',
            activation=None)
        if verbose: print(net)
        # Residual function (identity shortcut)
        net = tf.add(x_orig, net)
        if verbose: print('end of resnet\n', net)
        return net

    ############################################################
    # With blocks and bottlenecks.
    if verbose: print(x_orig)
    with tf.variable_scope('conv_layer1'):
        net = layers.conv2d(
            x_orig, 
            filters=num_filters,
            kernel_size=filter_size,
            padding='same',
            activation=tf.nn.relu)
        net = layers.batch_normalization(net, training=training)
        if verbose: print(net)
    if do_maxpool:
        net = tf.layers.max_pooling2d(
            net, 
            pool_size=3,
            strides=2,
            padding='same')
        if verbose: print(net)
    # Chain of convnets.
    with tf.variable_scope('conv_layer2'):
        net = tf.layers.conv2d(
            net,
            filters=128,
            kernel_size=1,
            padding='valid')
        if verbose: print(net)
    with tf.variable_scope('group1/block1/conv_in'):
        conv = tf.layers.conv2d(
            net,
            filters=128,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)
        if verbose: print(conv)
    with tf.variable_scope('group1/block1/conv_bottleneck'):
        conv = tf.layers.conv2d(
            conv,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)
        if verbose: print(conv)
    # 1x1 convolution responsible for restoring dimension
    with tf.variable_scope('group1/block1/conv_out'):
        input_dim = net.get_shape()[-1].value
        conv = tf.layers.conv2d(
            conv,
            filters=input_dim,
            kernel_size=1,
            padding='valid',
            activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=training)
        if verbose: print(conv)

    # shortcut connections that turn the network into its counterpart
    # residual function (identity shortcut)
    net = conv + net
    if verbose: print('end of resnet\n', net)
    return net


def RESNET_weights_from_images(
        x, base_size, scale_size, input_channel, repeat_num, num_filters,
        filter_size, data_format, reuse, dropout_pr=1.0, use_bias=False,
        training=True, verbose=False):
    """Maps (batch_size, scale_size, scale_size, 3) to weight prediction."""
    Group = namedtuple('Group', ['num_blocks', 'num_filters'])
    groups = [Group(1, 32), Group(1, 64), Group(0, 256)]

    verbose = True 
    if verbose:
        print('\n\nRESNET ARCHITECTURE\n')
        print(x)

    with tf.variable_scope('RESNET', reuse=reuse) as vs_resnet:
        # First conv.
        x = layers.conv2d(
            x, 
            filters=groups[0].num_filters,
            kernel_size=7,
            padding='same',
            activation=tf.nn.relu)
        if verbose: print(x)
        # Max pool, half the size.
        x = tf.layers.max_pooling2d(
            x, 
            pool_size=3,
            strides=2,
            padding='same')
        if verbose: print(x)
        # Layers of 64 filters, then half the input size.
        for _ in range(groups[0].num_blocks):
            x = resnet_conv2d(x, groups[0].num_filters, 3, training=training,
                              verbose=verbose)
        if verbose: print(x)
        x = tf.layers.max_pooling2d(
            x, 
            pool_size=3,
            strides=2,
            padding='same')
        if verbose: print(x)
        # Layers of 128 filters, then half the input size.
        if groups[1].num_blocks > 0:
            x = tf.layers.conv2d(
                x,
                filters=groups[1].num_filters,
                kernel_size=1,
                padding='same',
                activation=None,
                bias_initializer=None)
        for _ in range(groups[1].num_blocks):
            x = resnet_conv2d(x, groups[1].num_filters, 3, training=training,
                              verbose=verbose)
        if verbose: print(x)
        x = tf.layers.max_pooling2d(
            x, 
            pool_size=3,
            strides=2,
            padding='same')
        if verbose: print(x)
        # Layers of 256 filters, then half the input size.
        if groups[2].num_blocks > 0:
            x = tf.layers.conv2d(
                x,
                filters=groups[2].num_filters,
                kernel_size=1,
                padding='same',
                activation=None,
                bias_initializer=None)
        for _ in range(groups[2].num_blocks):
            x = resnet_conv2d(x, groups[2].num_filters, 3, training=training,
                              verbose=verbose)
        if verbose: print(x)
        #x = tf.layers.max_pooling2d(
        #    x, 
        #    pool_size=3,
        #    strides=2,
        #    padding='same')
        #if verbose: print(x)

        # Average pool at the end.
        x_shape = x.get_shape().as_list()
        x = tf.nn.avg_pool(
            x,
            ksize=[1, x_shape[1], x_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        if verbose: print(x)

        # Final fully connected layer to weight prediction.
        final_conv_flat_dim = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, final_conv_flat_dim])
        if verbose: print(x)

        x = layers.dense(x, 500)
        x = tf.nn.dropout(x, dropout_pr)
        if verbose: print(x)
        predictions = layers.dense(x, 1)
        if verbose: print(predictions)

    variables = tf.contrib.framework.get_variables(vs_resnet)
    return predictions, variables 


def predict_weights_from_images(
        x, base_size, scale_size, input_channel, repeat_num, num_filters,
        filter_size, data_format, reuse, dropout_pr=1.0, use_bias=False,
        verbose=False):
    """Maps (batch_size, scale_size, scale_size, 3) to weight prediction."""

    verbose = True 
    if verbose:
        print('\n\nimgs2wts ARCHITECTURE\n')
        print(x)

    channels_out = x.shape.as_list()[-1]
    current_num_filters = 2 ** (int(np.log2(num_filters)) - repeat_num)

    with tf.variable_scope('img2wts', reuse=reuse) as vs_img2wts:
        # like encoder. 
        current_dim = scale_size
        for idx in range(repeat_num + 1):
            x = conv2d(x, current_num_filters, filter_size, use_bias=use_bias,
                batch_resid=False, extra_dense=False, resize_scale=0.5)
            current_dim /= 2
            current_num_filters *= 2

            if verbose: print(x)

        final_conv_flat_dim = np.prod(x.shape.as_list()[1:])
        x = tf.reshape(x, [-1, final_conv_flat_dim])
        if verbose: print(x)

        x = tf.nn.dropout(x, dropout_pr)
        x = layers.dense(x, 64, activation=tf.nn.elu)
        x = tf.nn.dropout(x, dropout_pr)
        if verbose: print(x)
        x = layers.dense(x, 8, activation=tf.nn.elu)
        if verbose: print(x)
        #x = layers.dense(x, 256, activation=tf.nn.elu)
        #x = tf.nn.dropout(x, dropout_pr)
        #if verbose: print(x)
        #x = layers.dense(x, 128, activation=tf.nn.elu)
        #x = tf.nn.dropout(x, dropout_pr)
        #if verbose: print(x)
        #x = layers.dense(x, 64, activation=tf.nn.elu)
        #x = tf.nn.dropout(x, dropout_pr)
        #if verbose: print(x)
        #x = layers.dense(x, 32, activation=tf.nn.elu)
        #x = tf.nn.dropout(x, dropout_pr)
        #if verbose: print(x)
        #x = layers.dense(x, 16, activation=tf.nn.elu)
        #x = tf.nn.dropout(x, dropout_pr)
        #if verbose: print(x)

        predictions = layers.dense(x, 1)
        if verbose: print(predictions)

        variables = tf.contrib.framework.get_variables(vs_img2wts)
        return predictions, variables 


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
    return mmd


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

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def resize_(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)


def conv2d_(x, W):
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
    act = activation_choice
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
