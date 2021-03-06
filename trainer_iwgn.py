from __future__ import print_function

import os
import pdb
import StringIO
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/maurice/mmd')

from glob import glob
from itertools import chain
from PIL import Image
from scipy.stats import ks_2samp
from tqdm import trange

from data_loader import get_loader
from models import *
from mmd_utils import MMD_vs_Normal_by_filter
from utils import save_image

np.random.seed(123)


def next(loader):
    return loader.next()[0].data.numpy()


def vert(arr):
    return np.reshape(arr, [-1, 1])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


def sum_normed(mat):
    return mat / tf.reduce_sum(mat)


def to_nhwc(image, data_format, is_tf=False):
    if is_tf:
        if data_format == 'NCHW':
            new_image = nchw_to_nhwc(image)
        else:
            new_image = image
        return new_image
    else:
        if data_format == 'NCHW':
            new_image = image.transpose([0, 2, 3, 1])
        else:
            new_image = image
        return new_image


def nhwc_to_nchw(image, is_tf=False):
    if is_tf:
        if image.get_shape().as_list()[3] in [1, 3]:
            new_image = tf.transpose(image, [0, 3, 1, 2])
        else:
            new_image = image
        return new_image
    else:
        if image.shape[3] in [1, 3]:
            new_image = image.transpose([0, 3, 1, 2])
        else:
            new_image = image
        return new_image


def convert_255_to_n11(image):
    ''' Converts pixel values to range [-1, 1].'''
    image = image/127.5 - 1.
    return image


def convert_n11_to_255(image, is_tf=False):
    if is_tf:
        return tf.clip_by_value((image + 1)*127.5, 0, 255)
    else:
        return np.clip((image + 1)*127.5, 0, 255)


def convert_01_to_n11(image):
    return image * 2. - 1.


def convert_01_to_255(image, data_format):
    return np.clip(image * 255, 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def sort_and_scale(arr):
    """Returns CDF style normalization of array."""
    assert len(arr.shape) == 1, 'Array must be one-dimensional.'
    left_bounded_at_zero = arr - np.min(arr) 
    sorted_arr = np.sort(left_bounded_at_zero)
    sorted_and_scaled_zero_to_one = sorted_arr / (np.max(sorted_arr) + 1e-7)
    return sorted_and_scaled_zero_to_one

class Trainer(object):
    def __init__(self, config, data_loader, images_user, images_user_weights):
        self.config = config
        self.split = config.split
        #self.data_path = config.data_path
        self.data_loader = data_loader
        self.images_user = images_user
        self.images_user_weights = images_user_weights
        self.user_weights_min = np.min(self.images_user_weights)
        self.dataset = config.dataset
        self.mnist_class = config.mnist_class

        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.use_mmd= config.use_mmd
        self.lambda_mmd_setting = config.lambda_mmd_setting
        self.lambda_ae_setting = config.lambda_ae_setting
        self.weighted_setting = config.weighted

        self.step = tf.Variable(0, name='step', trainable=False)

        self.d_lr = tf.Variable(config.d_lr, name='d_lr', trainable=False)
        self.g_lr = tf.Variable(config.g_lr, name='g_lr', trainable=False)
        self.w_lr = tf.Variable(config.w_lr, name='w_lr', trainable=False)
        self.d_lr_update = tf.assign(
            self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
            name='d_lr_update')
        self.g_lr_update = tf.assign(
            self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
            name='g_lr_update')
        self.w_lr_update = tf.assign(
            self.w_lr, tf.maximum(self.w_lr * 0.5, config.lr_lower_boundary),
            name='w_lr_update')

        self.z_dim = config.z_dim
        self.num_conv_filters = config.num_conv_filters
        self.filter_size = config.filter_size
        self.use_bias = config.use_bias

        #self.model_dir = config.model_dir
        #self.load_path = config.load_path
        self.log_dir = config.log_dir

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, self.height, self.width, self.channel = \
            get_conv_shape(self.data_loader, self.data_format)
        self.scale_size = self.height 
        self.base_size = config.base_size
        log2_scale_size = int(np.log2(self.scale_size))
        log2_base_size = int(np.log2(self.base_size))
        # Convolutions from 64 down to base_size 4, and 2^(6-2), so 4 conv2d's.
        # where last one is separated out, so 3 repeated convolutions.
        self.repeat_num = log2_scale_size - log2_base_size - 1

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_dir)

        sv = tf.train.Supervisor(logdir=self.log_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False


    # Set up read-only nodes.
    def build_real_only(self):
        # Encode.
        self.to_encode_readonly = tf.placeholder(tf.float32,
            name='to_encode_readonly',
            shape=[None, self.scale_size, self.scale_size, self.channel])
        #to_enc = nhwc_to_nchw(convert_255_to_n11(self.to_encode_readonly), is_tf=True)
        #to_enc = convert_255_to_n11(self.to_encode_readonly)
        to_enc = self.to_encode_readonly
        _, self.encoded_readonly, _, _ = AutoencoderCNN(
            to_enc, self.base_size, self.scale_size, self.channel, self.z_dim,
            self.repeat_num, self.num_conv_filters, self.filter_size,
            self.data_format, reuse=True, use_bias=self.use_bias)

        # Decode.
        self.to_decode_readonly = tf.placeholder(tf.float32,
            shape=[None, self.z_dim], name='to_decode_readonly',)
        self.decoded_readonly, _, _, _ = AutoencoderCNN(
            to_enc, self.base_size, self.scale_size, self.channel, self.z_dim,
            self.repeat_num, self.num_conv_filters, self.filter_size,
            self.data_format, reuse=True, to_decode=self.to_decode_readonly,
            use_bias=self.use_bias)

        # Generate.
        self.z_read = tf.placeholder(tf.float32, shape=[None, self.z_dim],
            name='z_read',)
        g_read, _ = GeneratorCNN(
            self.z_read, self.base_size, self.num_conv_filters,
            self.filter_size, self.channel, self.repeat_num, self.data_format,
            reuse=True, use_bias=self.use_bias)
        self.g_read = convert_n11_to_255(g_read, is_tf=True)


    def build_model(self):
        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])
        # Images are NHWC on [0, 255].
        self.x = self.data_loader  # nhwc on [0,255]
        self.x_predicted_weights = tf.placeholder(tf.float32, [self.batch_size, 1],
            name='x_predicted_weights')
        x = convert_255_to_n11(self.x)
        self.weighted = tf.placeholder(tf.bool, name='weighted')

        # Set up generator and autoencoder functions. Output g is on [-1,1].
        g, self.g_var = GeneratorCNN(
            self.z, self.base_size, self.num_conv_filters, self.filter_size,
            self.channel, self.repeat_num, self.data_format, reuse=False,
            use_bias=self.use_bias, verbose=True)
        d_out, d_enc, self.d_var_enc, self.d_var_dec = AutoencoderCNN(
            tf.concat([x, g], 0), self.base_size, self.scale_size, self.channel,
            self.z_dim, self.repeat_num, self.num_conv_filters,
            self.filter_size, self.data_format, reuse=False,
            use_bias=self.use_bias, verbose=True)
        AE_x, AE_g = tf.split(d_out, 2)
        self.x_enc, self.g_enc = tf.split(d_enc, 2)
        self.g = convert_n11_to_255(g, is_tf=True)
        self.AE_g = convert_n11_to_255(AE_g, is_tf=True)
        self.AE_x = convert_n11_to_255(AE_x, is_tf=True)

        self.build_real_only()

        # Set up discriminator output.
        D_logits_out, self.D_var = DiscriminatorCNN(
            tf.concat([x, g], 0), self.base_size, self.scale_size, self.channel,
            self.repeat_num, self.num_conv_filters, self.filter_size,
            self.data_format, reuse=False, use_bias=self.use_bias, verbose=True)
        D_logits_x, D_logits_g = tf.split(D_logits_out, 2)
        self.D_loss_x = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logits_x, labels=tf.ones_like(D_logits_x)))
        self.D_loss_g = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logits_g, labels=tf.zeros_like(D_logits_g)))


        ## BEGIN: MMD DEFINITON
        on_encodings = 1
        if on_encodings:
            # Kernel on encodings.
            self.xe = self.x_enc 
            self.ge = self.g_enc 
            #sigma_list = [1., 2., 4., 8., 16.]
            sigma_list = [0.001, 0.01, 0.1, 1., 2.]
            #sigma_list = [0.001, 0.01, 0.1, 0.5, 1.]
            #sigma_list = [0.1, 0.5, 1., 2., 4., 8.]
            #sigma_list = [0.1, 0.5, 1.]
        else:
            # Kernel on full imgs.
            self.xe = tf.reshape(x, [tf.shape(x)[0], -1]) 
            self.ge = tf.reshape(g, [tf.shape(g)[0], -1]) 
            sigma_list = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        data_num = tf.shape(self.xe)[0]
        gen_num = tf.shape(self.ge)[0]
        v = tf.concat([self.xe, self.ge], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, [1, tf.shape(sqs)[0]])
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        self.K = K
        K_xx = K[:data_num, data_num:]
        K_yy = K[data_num:, data_num:]
        K_xy = K[:data_num, data_num:]
        K_xx_upper = upper(K_xx)
        K_yy_upper = upper(K_yy)
        num_combos_xx = tf.to_float(data_num * (data_num - 1) / 2)
        num_combos_yy = tf.to_float(gen_num * (gen_num - 1) / 2)

        # Build weights prediction.
        self.build_weights_prediction()

        self.x_predicted_weights_tiled = tf.tile(self.x_predicted_weights,
            [1, self.batch_size])
        # Autoencoder weights.
        self.p1_weights_ae = self.x_predicted_weights_tiled
        self.p1_weights_ae_normed = sum_normed(self.p1_weights_ae)
        # MMD weights.
        self.p1_weights = self.x_predicted_weights_tiled
        self.p1_weights_normed = sum_normed(self.p1_weights)
        self.p1p2_weights = self.p1_weights * tf.transpose(self.p1_weights)
        self.p1p2_weights_upper = upper(self.p1p2_weights)
        self.p1p2_weights_upper_normed = sum_normed(self.p1p2_weights_upper)
        Kw_xx_upper = K_xx_upper * self.p1p2_weights_upper_normed
        Kw_xy = K_xy * self.p1_weights_normed
        num_combos_xy = tf.to_float(data_num * gen_num)

        # Compute and choose between MMD values.
        self.mmd2 = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(Kw_xx_upper) +
                tf.reduce_sum(K_yy_upper) / num_combos_yy -
                2 * tf.reduce_sum(Kw_xy)),
            lambda: (
                tf.reduce_sum(K_xx_upper) / num_combos_xx +
                tf.reduce_sum(K_yy_upper) / num_combos_yy -
                2 * tf.reduce_sum(K_xy) / num_combos_xy))
        self.mmd2_weighted = (
            tf.reduce_sum(Kw_xx_upper) +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(Kw_xy))
        self.mmd2_unweighted = (
            tf.reduce_sum(K_xx_upper) / num_combos_xx +
            tf.reduce_sum(K_yy_upper) / num_combos_yy -
            2 * tf.reduce_sum(K_xy) / num_combos_xy)
        ## END: MMD DEFINITON


        ##########################################
        # DEFINE LOSSES.

        self.lambda_mmd = tf.Variable(0., trainable=False, name='lambda_mmd')
        self.lambda_ae = tf.Variable(0., trainable=False, name='lambda_ae')

        # AUTOENCODER
        self.ae_loss_real = tf.cond(
            self.weighted,
            lambda: (
                tf.reduce_sum(self.p1_weights_ae_normed * tf.reshape(
                    tf.reduce_sum(tf.square(AE_x - x), [1, 2, 3]), [-1, 1]))),
            lambda: tf.reduce_mean(tf.square(AE_x - x)))
        self.ae_loss_fake = tf.reduce_mean(tf.square(AE_g - g))
        #self.ae_loss = self.ae_loss_real + self.ae_loss_fake
        self.ae_loss = self.ae_loss_real

        # ENCODER NORM
        # Penalize norm above 1.
        #self.enc_norm_loss1 = tf.norm(self.x_enc)
        #self.enc_norm_loss1 = tf.norm(self.x_enc) + tf.norm(self.g_enc)
        self.enc_norm_loss1 = tf.nn.relu(tf.norm(self.x_enc) - 1.)
        #self.enc_norm_loss1 = tf.abs(1. - tf.norm(self.x_enc))  # Want norm approx 1.
        self.enc_norm_loss2 = (
            MMD_vs_Normal_by_filter(
                self.x_enc, np.ones([self.batch_size, 1], dtype=np.float32)))
        #self.enc_norm_loss2 = (
        #    MMD_vs_Normal_by_filter(
        #        self.x_enc, np.ones([self.batch_size, 1], dtype=np.float32)) +
        #    MMD_vs_Normal_by_filter(
        #        self.g_enc, np.ones([self.batch_size, 1], dtype=np.float32)))
        self.enc_norm_loss = 1. * self.enc_norm_loss1 + 1. * self.enc_norm_loss2


        # HINGE
        self.hinge_loss = tf.reduce_mean(tf.nn.relu(1. * 
            (tf.reduce_mean(self.x_enc, axis=0) - 
             tf.reduce_mean(self.g_enc, axis=0))))

        if self.dataset == 'mnist':
            self.d_loss = (self.lambda_ae * self.ae_loss + self.enc_norm_loss -
                self.lambda_mmd * self.mmd2) 
            self.g_loss = self.mmd2
        elif self.dataset == 'birds':
            self.d_loss = (self.lambda_ae * self.ae_loss -
                           self.lambda_mmd * self.mmd2 - 
                           self.hinge_loss)
            self.g_loss = self.mmd2 + self.hinge_loss
        elif self.dataset == 'celeba':
            #self.d_loss = (self.lambda_ae * self.ae_loss + self.enc_norm_loss -
            #               self.lambda_mmd * self.mmd2 - 
            #               16 * self.hinge_loss)
            self.d_loss = self.lambda_ae * self.ae_loss + 20. * self.enc_norm_loss
            self.g_loss = (self.lambda_mmd * self.mmd2 + 
                           16 * self.hinge_loss)
            #self.d_loss = (1. * self.ae_loss +
            #               self.D_loss_x - self.D_loss_g -
            #               0. * self.hinge_loss)
            #self.g_loss = self.D_loss_g + 0. * self.hinge_loss

        # Optimizer nodes.
        if self.optimizer == 'adam':
            ae_opt = tf.train.AdamOptimizer(self.d_lr)
            d_opt = tf.train.AdamOptimizer(self.d_lr)
            g_opt = tf.train.AdamOptimizer(self.g_lr)
        elif self.optimizer == 'rmsprop':
            ae_opt = tf.train.RMSPropOptimizer(self.d_lr)
            d_opt = tf.train.RMSPropOptimizer(self.d_lr)
            g_opt = tf.train.RMSPropOptimizer(self.g_lr)
        elif self.optimizer == 'sgd':
            ae_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            d_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            g_opt = tf.train.GradientDescentOptimizer(self.g_lr)


        # Set up optim nodes.
        clip = 1
        if clip:
            # CLIP ENCODER GRADIENTS.
            # Separately fetch encoder and decoder vars.
            enc_grads, enc_vars = zip(*d_opt.compute_gradients(
                self.d_loss, var_list=self.d_var_enc))
            dec_grads, dec_vars = zip(*d_opt.compute_gradients(
                self.d_loss, var_list=self.d_var_dec))
            # Clip encoder and decoder gradients.
            enc_grads_clipped = tuple(
                [tf.clip_by_value(g, -0.01, 0.01) for g in enc_grads])
            #dec_grads_clipped = tuple(
            #    [tf.clip_by_value(g, -0.01, 0.01) for g in dec_grads])
            # Reassemble list of gradients, and list of vars.
            d_grads = enc_grads_clipped + dec_grads
            d_vars = enc_vars + dec_vars
            self.d_optim = d_opt.apply_gradients(zip(d_grads, d_vars))

            # CLIP MMD GRADIENTS.
            #mmd_grads, mmd_vars = zip(*g_opt.compute_gradients(
            #    self.g_loss, var_list=self.g_var))
            #mmd_grads_clipped = tuple(
            #    [tf.clip_by_value(g, -0.01, 0.01) for g in mmd_grads])
            #self.g_optim = g_opt.apply_gradients(zip(
            #    mmd_grads_clipped, mmd_vars))
            self.g_optim = g_opt.minimize(
                self.g_loss, var_list=self.g_var, global_step=self.step)

            # TODO: TEST. TEMPORARILY TESTING ONLY AE_W
            self.d_optim = d_opt.apply_gradients(
                zip(d_grads, d_vars), global_step=self.step)

        else:
            ae_vars = self.d_var_enc + self.d_var_dec
            self.ae_optim = ae_opt.minimize(self.ae_loss_real, var_list=ae_vars)
            self.d_optim = d_opt.minimize(self.d_loss, var_list=ae_vars)
            self.g_optim = g_opt.minimize(
                self.g_loss, var_list=self.g_var, global_step=self.step)


        # SUMMARY
        # Sort real images by predicted weight.
        wts = tf.reshape(self.x_predicted_weights, [1, -1], name='wts')
        _, wts_order = tf.nn.top_k(wts, k=self.batch_size, name='wts_argsort')
        wts_order = tf.reshape(wts_order, [-1])
        self.x_sorted = tf.gather(self.x, wts_order, name='imgs_sorted')
        # Also, try random 10 from user, and logging them ordered.
        #random_ten = np.random.choice(400, 10)
        random_ten = tf.cast(tf.random_uniform(shape=(10,), minval=0, maxval=400), tf.int32)
        tu = tf.gather(self.w_images, random_ten)

        tu_wts = tf.reshape(tf.gather(self.w_weights, random_ten), [1, -1])
        _, tu_wts_order = tf.nn.top_k(tu_wts, k=10)
        tu_wts_order = tf.reshape(tu_wts_order, [-1])
        tu_sorted_wts = tf.gather(tu, tu_wts_order)

        tu_preds = tf.reshape(tf.gather(self.w_pred, random_ten), [1, -1])
        _, tu_preds_order = tf.nn.top_k(tu_preds, k=10)
        tu_preds_order = tf.reshape(tu_preds_order, [-1])
        tu_sorted_preds = tf.gather(tu, tu_preds_order)
        

        self.summary_op = tf.summary.merge([
            tf.summary.image("imgs_sorted", self.x_sorted, max_outputs=64),
            tf.summary.image("tu_sorted_wts", tu_sorted_wts, max_outputs=10),
            tf.summary.image("tu_sorted_preds", tu_sorted_preds, max_outputs=10),
            tf.summary.image("a_g", self.g, max_outputs=10),
            tf.summary.image("b_AE_g", self.AE_g, max_outputs=10),
            tf.summary.image("c_x", self.x, max_outputs=10),
            tf.summary.image("d_AE_x", self.AE_x, max_outputs=10),
            tf.summary.scalar("loss/w_loss", self.w_loss),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/ae_loss_real", self.ae_loss_real),
            tf.summary.scalar("loss/ae_loss_fake", self.ae_loss_fake),
            tf.summary.scalar("loss/enc_norm_loss1", self.enc_norm_loss1),
            tf.summary.scalar("loss/enc_norm_loss2", self.enc_norm_loss2),
            tf.summary.scalar("loss/hinge_loss", self.hinge_loss),
            tf.summary.scalar("loss/mmd2_u", self.mmd2_unweighted),
            tf.summary.scalar("loss/mmd2_w", self.mmd2_weighted),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

        
    def build_weights_prediction(self):
        # Images are NHWC on [0, 255].
        self.w_images = tf.placeholder(tf.float32,
            [None, self.scale_size, self.scale_size, self.channel], name='w_images')
        self.w_weights = tf.placeholder(tf.float32, [None, 1], name='w_weights')
        self.dropout_pr = tf.placeholder(tf.float32, name='dropout_pr')
        self.w_training = tf.placeholder(tf.bool, name='w_training_bool')

        # Convert NHWC [0, 255] to NCHW [-1, 1] for autoencoder.
        #w_images_for_ae = convert_255_to_n11(nhwc_to_nchw(self.w_images, is_tf=True))
        #w_images_for_ae = convert_255_to_n11(self.w_images)
        w_images_for_ae = self.w_images
        _, w_enc, _, _ = AutoencoderCNN(
            w_images_for_ae, self.base_size, self.scale_size, self.channel,
            self.z_dim, self.repeat_num, self.num_conv_filters,
            self.filter_size, self.data_format, reuse=True,
            use_bias=self.use_bias)


        #self.w_pred, self.w_vars = predict_weights_from_enc(
        #    w_enc, self.dropout_pr, reuse=False)
        #self.w_pred, self.w_vars = predict_weights_from_images(
        #    self.w_images, self.base_size, self.scale_size, self.channel,
        #    self.repeat_num, self.num_conv_filters, self.filter_size,
        #    self.data_format, reuse=False, dropout_pr=self.dropout_pr,
        #    use_bias=self.use_bias, verbose=True)
        self.w_pred, self.w_vars = RESNET_weights_from_images(
            self.w_images, self.base_size, self.scale_size, self.channel,
            self.repeat_num, self.num_conv_filters, self.filter_size,
            self.data_format, reuse=False, dropout_pr=self.dropout_pr,
            use_bias=self.use_bias, training=self.w_training, verbose=True)

        #self.w_loss = tf.reduce_mean(tf.squared_difference(self.w_weights, self.w_pred))
        #self.w_loss = tf.reduce_mean(tf.pow(tf.abs(self.w_weights - self.w_pred), 5))
        self.w_loss = tf.norm(tf.abs(self.w_weights - self.w_pred), ord=3)

        # Define optimization procedure.
        if 1:
            self.w_optim = tf.train.RMSPropOptimizer(self.w_lr).minimize(
                self.w_loss, var_list=self.w_vars)
        else:
            # Same as  above, but with gradient clipping.
            w_opt = tf.train.RMSPropOptimizer(self.w_lr)
            w_gvs = w_opt.compute_gradients(
                self.w_loss, var_list=self.w_vars)
            w_capped_gvs = (
                [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in w_gvs])
            self.w_optim = w_opt.apply_gradients(w_capped_gvs) 


    def predict_weights(self, inputs):
        # Inputs are NHWC on [0, 255].
        weights = self.sess.run(self.w_pred,
            feed_dict={
                self.w_images: inputs,
                self.dropout_pr: 1.0,
                self.w_training: False})
        # Trim low weights to the minimum among the user weights.
        weights[weights < self.user_weights_min] = self.user_weights_min
        return weights


    def generate(self, inputs, root_path=None, step=None, save=False):
        x = self.sess.run(self.g_read, {self.z_read: inputs})
        if save:
            path = os.path.join(root_path, 'G_{}.png'.format(step))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def encode(self, inputs):
        return self.sess.run(self.encoded_readonly, {self.to_encode_readonly: inputs})


    def decode(self, inputs):
        out = self.sess.run(self.decoded_readonly, {self.to_decode_readonly: inputs})
        out_unnormed = convert_n11_to_255(out)
        return out_unnormed


    def get_images_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def prep_data(self, split='user', n=100):
        # NOTE: This must correspond to the target_num_user_weights.npy file.
        target_num = self.mnist_class
        print('\n\nFetching only number {}.\n\n'.format(target_num))

        def fetch_and_prep(zipped_images_and_labels):
            d = zipped_images_and_labels
            # Each element of d is a list of [image, one-hot label].
            ind = [i for i,v in enumerate(d) if v[1][target_num] == 1]
            d_target_num = [v for i,v in enumerate(d) if i in ind]
            if split != 'user':
                d_target_num = np.random.permutation(d_target_num)
            images = [v[0] for v in d_target_num[:n]]
            labels = [v[1] for v in d_target_num[:n]]

            # Reshape, rescale, recode.
            images = np.reshape(images,
                [len(images), self.scale_size, self.scale_size, self.channel])
            images = convert_01_to_255(images)
            return images

        m = self.mnist
        if split == 'user':
            imgs_and_labs = zip(m.validation.images, m.validation.labels)
            images = fetch_and_prep(imgs_and_labs)
        elif split == 'train':
            imgs_and_labs = zip(m.train.images, m.train.labels)
            images = fetch_and_prep(imgs_and_labs)
        elif split == 'test':
            imgs_and_labs = zip(m.test.images, m.test.labels)
            images = fetch_and_prep(imgs_and_labs)

        return images


    def get_n_images_and_weights(self, n, images, weights):
        """Randomly samples imgs and their weights, for user-labeled set."""
        assert n <= len(images), 'n must be less than length of image set'
        n_random_indices = np.random.choice(range(len(images)), n, replace=False)
        n_images = [v for i,v in enumerate(images) if i in n_random_indices]
        n_weights = [v for i,v in enumerate(weights) if i in n_random_indices]
        return np.array(n_images), vert(n_weights)
    

    def get_n_images(self, n, images):
        """Randomly samples from a given set of images."""
        assert n <= len(images), 'n must be less than length of image set'
        n_random_indices = np.random.choice(range(len(images)), n, replace=False)
        n_images = [v for i,v in enumerate(images) if i in n_random_indices]
        return np.array(n_images)


    def interpolate_z(self, step, batch_train):
        # Interpolate z when sampled from noise.
        z1 = np.random.normal(0, 1, size=(1, self.z_dim))
        z2 = np.random.normal(0, 1, size=(1, self.z_dim))
        num_interps = 10
        proportions = np.linspace(0, 1, num=num_interps)
        zs = np.zeros(shape=(num_interps, self.z_dim))
        gens = np.zeros([num_interps, self.scale_size, self.scale_size, self.channel])
        for i in range(num_interps):
            zs[i] = proportions[i] * z1 + (1 - proportions[i]) * z2
            gens[i] = self.generate(np.reshape(zs[i], [1, -1]))
        save_image(gens, '{}/interpolate_z_noise{}.png'.format(
            self.log_dir, step))

        # Interpolate between two random encodings.
        #two_random_images = self.get_n_images(2, self.images_train)
        two_random_images = batch_train[:2]
        im1 = two_random_images[:1, :, :, :]  # This notation keeps dims.
        im2 = two_random_images[1:, :, :, :]
        z1 = self.encode(im1)
        z2 = self.encode(im2)
        num_interps = 10
        proportions = np.linspace(0, 1, num=num_interps)
        zs = np.zeros(shape=(num_interps, self.z_dim))
        gens = np.zeros([num_interps, self.scale_size, self.scale_size, self.channel])
        for i in range(num_interps):
            zs[i] = proportions[i] * z1 + (1 - proportions[i]) * z2
            gens[i] = self.generate(np.reshape(zs[i], [1, -1]))
        save_image(gens, '{}/interpolate_z_enc{}.png'.format(
            self.log_dir, step))


    def save_sorted_images(self, step, images):
        """Plot images sorted by predicted weight, and show distributions."""
        # Get data and their predicted weights, and sort images by weight.
        weights = self.predict_weights(images).flatten()
        wt_order = np.argsort(weights)
        images_sorted = images[wt_order]
        weights_sorted = weights[wt_order]
        #print(weights_sorted)
        save_image(images_sorted, '{}/wt_sorted_data{}.png'.format(
            self.log_dir, step), nrow=int(np.sqrt(self.batch_size)))


    def show_sorted_images(self, step, batch_train):
        """Plot images sorted by predicted weight, and show distributions."""
        # Get data and their predicted weights, and sort images by weight.
        data = batch_train
        data_weights = self.predict_weights(data).flatten()
        data_sorted = data[np.argsort(data_weights)]

        # Get gens and their predicted weights, and sort images by weight.
        zs = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        gens = self.generate(zs)
        gens_weights = self.predict_weights(gens).flatten()
        gens_sorted = gens[np.argsort(gens_weights)]

        # Save the sorted versions of data and gens images.
        save_image(gens_sorted, '{}/sorted_gens{}.png'.format(
            self.log_dir, step), nrow=int(np.sqrt(self.batch_size)))
        save_image(data_sorted, '{}/sorted_data{}.png'.format(
            self.log_dir, step), nrow=int(np.sqrt(self.batch_size)))
        

        # CREATE UPSAMPLED VERSIONS FOR LINE PLOTS. ###########################
        # Get upsample weights for data.
        #num_to_lineplot = 100
        #d = self.get_n_images(num_to_lineplot, self.images_train)
        d = batch_train
        d_weights = self.predict_weights(d).flatten()
        d_weights_up = []
        for weight in d_weights:
            # Add the weight once.
            d_weights_up.append(weight)
            rounded_weight = int(np.round(weight))
            if rounded_weight > 1:
                # Add copies of the weight, according to rounded weight val.
                for _ in range(rounded_weight - 1):
                    d_weights_up.append(weight)
        d_weights_up = np.array(d_weights_up)

        # Make large set of gens to plot.
        zs_ = np.random.normal(0, 1, size=(len(d_weights_up), self.z_dim))
        g = self.generate(zs_)
        g_weights = self.predict_weights(g).flatten()

        # Scale the sorted weights to [0,1].
        #trim = 0  # TODO: Trim because weight predictor gives very low weights for some gens.
        #d = np.sort(d_weights)[trim:-trim]
        #d_up = np.sort(d_weights_up)[trim:-trim]
        #g_up = np.sort(g_weights_up)[trim:-trim]

        d_plot = sort_and_scale(d_weights)
        d_plot_up = sort_and_scale(d_weights_up)
        g_plot = sort_and_scale(g_weights)

        # Compute Kolmogorov-Smirnov distance between upsampled data weights,
        #   and gens weights.
        ks_dist, ks_pval = ks_2samp(g_weights, d_weights_up)
        ks_dist_, ks_pval_ = ks_2samp(g_weights, d_weights)
        print('KS(gens, up). dist={:.2f}, p={:.3f}'.format(ks_dist, ks_pval))
        print('KS(gens, data). dist={:.2f}, p={:.3f}'.format(ks_dist_, ks_pval_))


        # PLOT SORTED WEIGHTS. ################################################
        # Make line graph to show differences of distributions.
        plt.figure(figsize=(5, 5))


        sparser_plot = 0
        if sparser_plot:
            plt.plot(np.linspace(0, 1, len(d_plot[::30])), d_plot[::30], color='gray', marker='.', label='data')
            plt.plot(np.linspace(0, 1, len(d_plot_up[::30])), d_plot_up[::30], color='red', marker='*', label='up')
            plt.plot(np.linspace(0, 1, len(g_plot[::30])), g_plot[::30], color='green', marker='+', label='gens')
            plt.xlabel('Ordered sample')
            plt.ylabel('CDF, predicted weight')
            plt.legend()
            plt.title('KS(gens, up). dist={:.2f}, p={:.3f}'.format(ks_dist, ks_pval))
            plt.subplots_adjust(bottom=.25, left=.25)
            plt.savefig('{}/sorted_lines{}.png'.format(self.log_dir, step))
            plt.legend()
            plt.close()
        else:
            plt.plot(np.linspace(0, 1, len(d_plot)), d_plot, color='gray', marker='.', label='data')
            plt.plot(np.linspace(0, 1, len(d_plot_up)), d_plot_up, color='red', marker='*', label='up')
            plt.plot(np.linspace(0, 1, len(g_plot)), g_plot, color='green', marker='+', label='gens')
            plt.xlabel('Ordered sample')
            plt.ylabel('CDF, predicted weight')
            plt.legend()
            plt.title('KS(gens, up). dist={:.2f}, p={:.3f}'.format(ks_dist, ks_pval))
            plt.subplots_adjust(bottom=.25, left=.25)
            plt.savefig('{}/sorted_lines{}.png'.format(self.log_dir, step))
            plt.legend()
            plt.close()




    def train(self):
        print('\n{}\n'.format(self.config))

        # Save some fixed images once.
        z_fixed = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
        x_fixed = self.get_images_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.log_dir))

        # One time, save images in order to apply ratings.
        user_imgs_dir = 'user_imgs'
        if not os.path.exists(user_imgs_dir):
            os.makedirs(user_imgs_dir)
        imgs = self.images_user
        save_image(self.images_user, '{}/user.png'.format(user_imgs_dir))
        #for i in range(1, len(self.images_user) + 1):
        #    im = Image.fromarray(imgs[i])
        #    im = im.convert('RGB')
        #    im.save('{}/user_{}.png'.format(user_imgs_dir, i))
        
        # Train generator.
        for step in trange(self.start_step, self.max_step):

            # WEIGHTS OPTIM.
            batch_user, batch_user_weights = self.get_n_images_and_weights(
                self.batch_size, self.images_user, self.images_user_weights)
            self.sess.run(self.w_optim,
                feed_dict={
                    self.w_images: batch_user,
                    self.w_weights: batch_user_weights,
                    self.dropout_pr: 0.5,
                    self.w_training: True})

            # D AND G OPTIMS.
            fetch_dict = {
                #'d_optim': self.d_optim,
                #'g_optim': self.g_optim,  # TODO: TEST. TEMPORARILY TESTING ONLY AE_W
            }
            schedule_optims = False 
            if schedule_optims:
                if step < 25 or step % 500 == 0:
                    # 100 d_optim for 1 g_optim
                    for i in range(1, 100):
                        fetch_dict.update({
                            'd_optim{}'.format(i): self.d_optim})
                else:
                    # 5 d_optim for 1 g_optim
                    for i in range(1, 5):
                        fetch_dict.update({
                            'd_optim{}'.format(i): self.d_optim})

            # Occasionally fetch other nodes for logging/saving.
            if step % self.log_step == 0:
                fetch_dict.update({
                    'summary': self.summary_op,
                    'w_loss': self.w_loss,
                    'ae_loss_real': self.ae_loss_real,
                    'ae_loss_fake': self.ae_loss_fake,
                    'enc_norm_loss': self.enc_norm_loss,
                    'enc_norm_loss1': self.enc_norm_loss1,
                    'enc_norm_loss2': self.enc_norm_loss2,
                    'hinge_loss': self.hinge_loss,
                    'mmd2': self.mmd2,
                    'x_enc': self.x_enc,
                    'g_enc': self.g_enc,
                    'w_weights': self.w_weights,
                    'w_pred': self.w_pred,
                })

            # For MMDGAN training, use data with predicted weights.
            #batch_train = self.get_n_images(self.batch_size, self.images_train)
            batch_train = self.sess.run(self.x)  # NHWC on [0,255]
            batch_train_weights = self.predict_weights(batch_train)

            # Run full training step on pre-fetched data and simulations.
            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.x: batch_train,
                    self.x_predicted_weights: batch_train_weights,
                    self.lambda_mmd: self.lambda_mmd_setting, 
                    self.lambda_ae: self.lambda_ae_setting,
                    self.w_images: self.images_user, #
                    self.w_weights: self.images_user_weights, #
                    self.dropout_pr: 1,
                    self.w_training: True,
                    self.weighted: self.weighted_setting})

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            # Log and save as needed.
            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                w_loss = result['w_loss']
                w_weights = result['w_weights']
                w_pred = result['w_pred']
                ae_loss_real = result['ae_loss_real']
                ae_loss_fake = result['ae_loss_fake']
                enc_norm_loss = result['enc_norm_loss']
                enc_norm_loss1 = result['enc_norm_loss1']
                enc_norm_loss2 = result['enc_norm_loss2']
                hinge_loss = result['hinge_loss']
                mmd2 = result['mmd2']
                print(('[{}/{}] LOSSES: ae_real/fake: {:.3f}, {:.3f} '
                    'mmd2: {:.3f}, hinge: {:.3f}, enc_norm: {:.3f} ({:.3f},'
                    '{:.3f}), w_loss: {:.3f}').format(
                        step, self.max_step, ae_loss_real, ae_loss_fake,
                        mmd2, hinge_loss, enc_norm_loss, enc_norm_loss1,
                        enc_norm_loss2, w_loss))


                # Troubleshoot predictions for reals with 5+ weight.
                true_w = np.reshape(w_weights, [1, -1])
                pred_w = np.reshape(w_pred, [1, -1])
                true_w_5m = np.where(true_w < 5)
                true_w_5p = np.where(true_w >= 5)

                # Perf on < 5.
                print('Perf on < 5')
                #print(np.round(true_w[true_w_5m], 1))
                #print(np.round(pred_w[true_w_5m], 1))
                print('  {}'.format(np.round(
                    np.percentile(true_w[true_w_5m],
                                  [0, 5, 20, 50, 80, 95, 100]), 1)))
                print('  {}'.format(np.round(
                    np.percentile(pred_w[true_w_5m],
                                  [0, 5, 20, 50, 80, 95, 100]), 1)))
                print('Perf on >= 5')
                #print(np.round(true_w[true_w_5p], 1))
                #print(np.round(pred_w[true_w_5p], 1))
                print('  {}'.format(np.round(
                    np.percentile(true_w[true_w_5p],
                                  [0, 5, 20, 50, 80, 95, 100]), 1)))
                print('  {}'.format(np.round(
                    np.percentile(pred_w[true_w_5p],
                                  [0, 5, 20, 50, 80, 95, 100]), 1)))
                print('w_loss: {:.3f}'.format(w_loss))


                # TROUBLESHOOT ENCODING RANGE.
                x_enc_ = result['x_enc']
                g_enc_ = result['g_enc']
                print(np.round(np.percentile(x_enc_, [0, 20, 50, 80, 100]), 2))
                #print(np.round(np.percentile(g_enc_, [0, 5, 20, 50, 80, 95, 100]), 3))

            if step % (self.save_step) == 0:
                # First save a sample.
                if step == 0:
                    #x_samp = self.get_n_images(1, self.images_train)
                    x_samp = batch_train[:1]  # This indexing keeps dims.
                    save_image(x_samp, '{}/x_samp.png'.format(self.log_dir))
                self.save_sorted_images(step, batch_train)

                # Save images for fixed and random z.
                gen_fixed = self.generate(
                    z_fixed, root_path=self.log_dir, step='fix'+str(step),
                    save=True)
                z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                gen_rand = self.generate(
                    z, root_path=self.log_dir, step='rand'+str(step),
                    save=True)

                # Save image of interpolation of z.
                self.interpolate_z(step, batch_train)

                # TODO: Resolve whether this was creating memory issues, where
                # batch size appeared to bloat.
                # Save 100 generated and 100 data, with increasing weight.
                #self.show_sorted_images(step, batch_train)

