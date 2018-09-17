import numpy as np
import os
import pdb
from glob import glob
from PIL import Image
from shutil import copyfile

"""
Below is useful for getting raw files from 'train' to 'user_raw'.
From inside the 'train' dir:
for file in $(ls -p | grep -v / | head -100)
do
mv $file ../user_raw
done
"""

# Get current directory.
cwd = os.getcwd()  # /home/maurice/iwgn_natimgs/data/CelebA/splits

# Create directory for resized images.
if not os.path.exists('user'):
    os.mkdir('user')

# Set up source directory.
path_user_raw = os.path.join(cwd, 'user_raw')

# Set up destination directory.
path_user = os.path.join(cwd, 'user')

# Fetch files from source.
paths = glob("{}/*.{}".format(path_user_raw, 'jpg'))

# For each path, open image, crop, resize, save to destination directory.
offset_height = 50
offset_width = 25
target_height = 128
target_width = 128
scale_size = 64
for path in paths:
    filename = path.split('/')[-1]
    img_raw = Image.open(path)
    # Crop is left, upper, right lower.
    img_cropped = img_raw.crop((offset_width, offset_height,
        offset_width + target_width, offset_height + target_height))
    img_resized = img_cropped.resize((scale_size, scale_size)) 
    img_resized.save(os.path.join(path_user, filename))


