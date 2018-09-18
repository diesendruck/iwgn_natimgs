import numpy as np
import os
import pdb
import tensorflow as tf
from PIL import Image
from glob import glob

from trainer_iwgn import nhwc_to_nchw


def get_loader(root, batch_size, scale_size, data_format, split_name=None,
        is_grayscale=False, seed=None, target=None):
    dataset_name = os.path.basename(root)
    if dataset_name == 'mnist':
        is_grayscale=True
        channels = 1
        scale_size = 28  # TODO: Determine whether scale should be 28.
    else:
        channels = 3
    
    if dataset_name in ['CelebA', 'mnist'] and split_name:
        if target:
            root = os.path.join(root, 'splits', target)
        else:
            root = os.path.join(root, 'splits', split_name)
    elif dataset_name in ['birds'] and split_name:
        root = os.path.join(root, split_name)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, channels]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=channels)
    # Test standardization here. 
    # Alternatively, do standardization in trainer_iwgn.py.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 2 * batch_size
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    elif dataset_name == 'birds':
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
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


def load_user(data_path, scale_size, data_format):
    user_path = os.path.join(data_path, 'splits', 'user')
    if not os.path.exists(user_path):
        os.mkdir(user_path)
        
    paths = glob("{}/*.{}".format(user_path, 'jpg'))
    assert len(paths) > 0, 'Did not find paths.'
    user_imgs = np.array([np.array(Image.open(path)) for path in paths])
    assert data_format == 'NHWC', 'data_format should be NHWC'
    return user_imgs 

