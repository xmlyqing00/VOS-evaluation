import os
import sys
import numpy as np
from PIL import Image
from glob import glob
import cv2



class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        
        img_list = np.sort(glob(os.path.join(self.root_dir, sequence, '[0-9]*.png'))).tolist()
        # if len(img_list) == 0:
            # img_list = np.sort(glob(os.path.join(self.root_dir, sequence, '*.jpg'))).tolist()
        # print(os.path.join(self.root_dir, sequence, '*.png'))
        # print(img_list)

        mask_0 = cv2.imread(img_list[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))

        for ii, m in enumerate(masks_id):
            masks[ii, ...] = cv2.imread(img_list[ii+1])
        masks[masks == 255] = 1
        masks = masks[..., 0]
        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks
