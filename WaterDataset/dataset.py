import os
from glob import glob
from collections import defaultdict
import numpy as np
import cv2


class WaterDataset(object):
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://google.com'
    VOID_LABEL = 255

    def __init__(self, root_folder, task='semi-supervised', sequences='all', resolution=(640, 480)):
        """
        Class to read the DAVIS dataset
        :param root_folder: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.root_folder = root_folder
        self.resolution = resolution
        self.img_path = os.path.join(self.root_folder, 'test_videos')
        self.mask_path = os.path.join(self.root_folder, 'test_annots')

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.root_folder, 'eval.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        try:
            sequences_names.remove('')
        except ValueError as e:
            pass
        print('Seq names:', sequences_names)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0:
                images = np.sort(glob(os.path.join(self.img_path, seq, '*.png'))).tolist()
                if len(images) == 0:
                    raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            self.sequences[seq]['masks'] = masks
        
    def _check_directories(self):
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(f'WaterDataset not found in the specified directory, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img_path in self.sequences[sequence]['images']:
            print(img_path)
            img = cv2.resize(cv2.imread(img_path), self.resolution)
            mask_path = img_path.replace('jpg', 'png').replace('test_videos', 'test_annots')
            print(mask_path)
            print(mask_path in self.sequences[sequence]['masks'])
            if mask_path in self.sequences[sequence]['masks']:
                mask = cv2.resize(cv2.imread(mask_path), self.resolution)
            else:
                mask = None
            yield img, mask

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

    def _get_all_elements(self, sequence, obj_type):
        obj = cv2.resize(cv2.imread(self.sequences[sequence][obj_type][0]), self.resolution)
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            tmp = cv2.resize(cv2.imread(obj), self.resolution)
            _, tmp = cv2.threshold(tmp, 128, 255, cv2.THRESH_BINARY)
            all_objs[i, ...] = tmp
            
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    # def get_all_images(self, sequence):
    #     return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=True):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks = masks[..., 0]

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks[i, masks[i, ...] == 255] = 1

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0

        return masks, masks_id


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = WaterDataset(root_folder='/Ship01/Dataset/water')
    for seq in dataset.get_sequences():
        g = dataset.get_frames(seq)
        for i in range(10):
            img, mask = next(g)
            plt.subplot(2, 1, 1)
            plt.title(seq)
            plt.imshow(img)
            plt.subplot(2, 1, 2)
            if mask is not None:
                plt.imshow(mask)
            plt.show(block=True)

