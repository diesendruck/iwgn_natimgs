import numpy as np
import os
import pdb
from glob import glob
from PIL import Image
from shutil import copyfile

"""
Below is useful for getting raw files from 'source' to 'dest'.
From inside the 'source' dir:
for file in $(ls -p | grep -v / | head -100)
do
cp $file dest 
done
"""


# Get current directory.
os.chdir('/home/maurice/iwgn_natimgs/data/birds')
cwd = os.getcwd()  # /home/maurice/iwgn_natimgs/data/birds
if not os.path.exists(os.path.join(cwd, 'images_preprocessed')):
    os.mkdir(os.path.join(cwd, 'images_preprocessed'))

# Build dict of dicts, to consolidate file info.
# {'id1': {'path': path_string, 'bbox': bbox_array, 'split': train_indicator},
#  'id2': ...
# }

# Get dict of {id: path}.
id_path_raw = open('images.txt', 'r').read().splitlines()
id_path = {}
for i in id_path_raw:
    _id, _path = i.split(' ')
    id_path[int(_id)] = _path

# Get dict of {id: bbox_array}.
id_bbox_raw = np.loadtxt('bounding_boxes.txt')  # Come as arrays.
id_bbox = {}
for row in id_bbox_raw:
    id_bbox[int(row[0])] = row[1:]

# Get dict of {id: train_indicator}.
id_istrain_raw = np.loadtxt('train_test_split.txt')  # Come as arrays.
id_istrain = {}
for row in id_istrain_raw:
    id_istrain[int(row[0])] = int(row[1])

# Make master dict.
id_info = {}
for _id in id_path.keys():
    id_info[_id] = {} 
    id_info[_id]['path'] = os.path.join(cwd, 'images', id_path[_id])
    id_info[_id]['bbox'] = id_bbox[_id]
    id_info[_id]['istrain'] = id_istrain[_id]
    id_info[_id]['filename'] = id_info[_id]['path'].split('/')[-1]

# For each image, load from path, process, and save to destination dir.
bboxed_sizes = np.zeros((len(id_info), 2))
for _id in id_info.keys():
    path = id_info[_id]['path']
    bbox = id_info[_id]['bbox']
    istrain = id_info[_id]['istrain']
    filename = id_info[_id]['filename']
    img_raw = Image.open(path)

    # Crop to bounding box.
    left, upper, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    img = img_raw.crop((left, upper, left + width, upper + height))

    # Pad with zeros for a square image.
    # 1. Get type of padding (width, heigh, none). Trim if pad dim is odd.
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    if h > w:
        pad = 'pad_width'
    elif w > h:
        pad = 'pad_height'
    elif h == w:
        pad = 'none'
    dim_diff = abs(h - w)
    if dim_diff % 2 != 0:
        h -= 1
    # 2. Get number of cols to pad.
    img = img[:h, :w]
    num_padding_cols = abs(h - w)
    # 3. Apply padding cols and save.
    if pad == 'pad_width':
        pad_cols = np.zeros((h, num_padding_cols/2, 3))
        try:
            img = np.concatenate((pad_cols, img, pad_cols) , axis=1)
        except:
            print('Skipped {} - {}'.format(_id, filename))
            continue
    elif pad == 'pad_height':
        pad_rows = np.zeros((num_padding_cols/2, w, 3))
        try:
            img = np.concatenate((pad_rows, img, pad_rows) , axis=0)
        except:
            print('Skipped {} - {}'.format(_id, filename))
            continue
    elif pad == 'none':
        pass
    
    # Save processed image.
    assert img.shape[0] == img.shape[1], 'image not square'
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(cwd, 'images_preprocessed', filename))

    
"""
(from images_preprocessed)
for file in $(ls -p | grep -v / | head -10000)
do
cp $file ../train 
done

for file in $(ls -p | grep -v / | head -200)
do
cp $file ../user 
done
"""
pdb.set_trace()
