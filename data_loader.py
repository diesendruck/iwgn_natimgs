import numpy as np
import os
import pdb
import sys
import tensorflow as tf
from glob import glob
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
sys.path.append('/home/maurice/global_utils')
from global_utils import natural_sort


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

    # Get user weights.
    user_weights = np.reshape(np.load('user_weights.npy'), [-1, 1])

    # Get user paths.
    assert os.path.exists(user_path), 'user_path does not exist'
    paths_loaded = glob("{}/*.{}".format(user_path, 'jpg'))
    paths = natural_sort(paths_loaded)
    assert len(paths) > 0, 'Did not find paths.'

    #resort_to_original_index = 0
    #if resort_to_original_index:
    #    paths_loaded_reorder = [paths.index(i) for i in paths_loaded]
    #    weights_loaded = np.load('user_weights_CELEBA.npy')
    #    weights_reordered = np.zeros(len(weights_loaded))
    #    for i in range(len(weights_loaded)):
    #        #weights_reordered[i] = weights_loaded[paths_loaded_reorder[i]]
    #        weights_reordered[paths_loaded_reorder[i]] = weights_loaded[i]
    #    np.save('user_weights_CELEBA_original_index.npy', weights_reordered)
    #    pdb.set_trace()

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
        augment = 1
        if augment:
            final_imgs = []
            final_weights = []
            for i, path in enumerate(paths):
                img_weight = user_weights[i]
                if img_weight >= 0:
                    # Data augmentation: mirrors, rotations, sharpening.
                    img = Image.open(path)
                    img_mirror = ImageOps.mirror(img)
                    img_rotate = img.rotate(np.random.randint(-20,20))
                    img_rotate2 = img.rotate(np.random.randint(-20,20))
                    img_mirror_rotate = img_mirror.rotate(np.random.randint(-20,20))
                    img_mirror_rotate2 = img_mirror.rotate(np.random.randint(-20,20))
                    img_sharp = img.filter(ImageFilter.UnsharpMask())
                    img_mirror_sharp = img_mirror.filter(ImageFilter.UnsharpMask())
                    img_rotate_sharp = img_rotate.filter(ImageFilter.UnsharpMask())
                    img_rotate2_sharp = img_rotate2.filter(ImageFilter.UnsharpMask())
                    img_mirror_rotate_sharp = img_mirror_rotate.filter(
                        ImageFilter.UnsharpMask())
                    img_mirror_rotate2_sharp = img_mirror_rotate2.filter(
                        ImageFilter.UnsharpMask())

                    image_set = [img, img_mirror,
                                 img_rotate, img_rotate2,
                                 img_mirror_rotate, img_mirror_rotate2]
                                 #img_sharp, img_mirror_sharp,
                                 #img_rotate_sharp, img_rotate2_sharp,
                                 #img_mirror_rotate_sharp, img_mirror_rotate2_sharp]

                    image_set = [np.array(m.resize((scale_size, scale_size),
                                                   Image.NEAREST)) \
                                for m in image_set]

                    final_imgs.extend(image_set)
                    final_weights.extend([img_weight] * len(image_set))
                else:
                    img = np.array(Image.open(path).resize((scale_size, scale_size),
                                   Image.NEAREST))
                    final_imgs.append(img)
                    final_weights.append(img_weight)

            user_imgs = np.array(final_imgs)
            user_weights = np.array(final_weights)
            assert user_imgs.shape[0] == user_weights.shape[0], \
                'img and wts mismatch'
            print user_weights.shape
            pdb.set_trace()

        else:
            user_imgs = np.array([
                np.array(Image.open(path).resize((scale_size, scale_size),
                    Image.NEAREST))
                for path in paths])
    assert data_format == 'NHWC', 'data_format should be NHWC'

    return user_imgs, user_weights

