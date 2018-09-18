import numpy as np
import os
import pdb
import sys
import tensorflow as tf

from trainer_iwgn import Trainer
from config import get_config
from data_loader import get_loader, load_user
from utils import prepare_dirs_and_logger, save_config

def main(config):
    # NOTE: Run this in shell first.
    #if tf.__version__[:3] != '1.1':
    #    sys.exit('***NOTE!***: FIRST RUN:\n"source ~/began/BEGAN-tensorflow/tf1.1/bin/activate"')
    # NOTE: Other setup requirements.
    print('\nREQUIREMENTS:\n  1. The file "user_weights.npy" should '
        'contain the user-provided labels for images in /user_images.\n')
    #print('Press "c" to continue.\n\n')
    #pdb.set_trace()

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    data_loader = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, split_name='train', grayscale=config.grayscale)
    images_user = load_user(config.dataset, data_path, config.scale_size, config.data_format,
        grayscale=config.grayscale)
    images_user_weights = np.load('user_weights.npy')
    trainer = Trainer(config, data_loader, images_user, images_user_weights)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
