import numpy as np
import os
import pdb
import tensorflow as tf
from PIL import Image
from glob import glob

from trainer_iwgn import nhwc_to_nchw


def get_loader(root, batch_size, scale_size, data_format, split_name=None,
        grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if dataset_name == 'mnist':
        grayscale=True
        channels = 1
        scale_size = 32  # TODO: Determine whether scale should be 28.
    else:
        if grayscale:
            channels = 1
        else:
            channels = 3
    
    root = os.path.join(root, split_name)  # for example, data/birds/train

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(paths) != 0:
            break
    assert len(paths) > 0, 'did not find paths'

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=channels)
    if dataset_name != 'mnist':
        image = tf.image.random_flip_left_right(image)  # Data augmentation.

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, channels]

    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
        image.set_shape([h, w, 1])
    else:
        image.set_shape(shape)

    min_after_dequeue = 2 * batch_size
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['celeba']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    elif dataset_name in ['birds']:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)


def load_user(dataset, data_path, scale_size, data_format, grayscale=False):
    user_path = os.path.join(data_path, 'user')
    assert os.path.exists(user_path), 'user_path does not exist'
        
    paths = glob("{}/*.{}".format(user_path, 'jpg'))
    assert len(paths) > 0, 'Did not find paths.'
    if dataset == 'mnist':
        grayscale = True
    if grayscale:
        user_imgs = np.array([
            np.expand_dims(
                np.array(Image.open(path).convert('L').resize(
                    (scale_size, scale_size), Image.BICUBIC)),
                axis=2)
            for path in paths])
    else:
        user_imgs = np.array([
            np.array(Image.open(path).resize((scale_size, scale_size),
                Image.NEAREST))
            for path in paths])
    assert data_format == 'NHWC', 'data_format should be NHWC'
    return user_imgs 

