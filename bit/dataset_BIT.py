"""
Date: 2019.01.20
Author: He Zhao
    Implement sample indice for BIT Flow branch:
        1. Cut off Each video record.num_frames into subset range of [20%, 80%], for removing the starting and ending;
        2. Segment new record.num_frames into new data length (i.e. 3 in my initial trail);
        3. Sample 5 index from each data segments;
        4. return 3 segments, in total 15 frames;
        5. Adding randomness into sampling process;

    Operation:
        1. compute video starting and ending index by record.num_frames * 20% and record.num_frames * 80%;
        2. compute average_duration = (ending - staring) // self.length // self.num_segs;
        3. return frame index for first segments and interval;
        4. return 5 frames for each of 3 segments;

    Update 2019.01.20:
        1. Achieves 67% for best;
        2. Next, try to achieves higher results by allow over-lapping sample for short-length videos; I.E. 29 length ==> 25 segments which requires 25 * 5 = 125 frames;
        3. Try Over-Lapping Sampling for testing first;

    Update 2019.01.21:
        1. Achieves 80.48% for best; The main reason, I suppose, is do random & over-lapping sampling for training process;
        2. For testing, especially for longer video, I think I need a more robust methods;
        3. For training, I would like to try using more segments, to see if make difference;

    Update 2019.01.22:
        1. Modify the code for sparser sampling for longer sequence video, so that information can be utilized more efficiently;
        2. That is: instead of using 1 interval, uniformly sample during segments;

    Update 2019.01.26:
        1. Dynamically adapt def _get_test_indices(), Flow, for i in range(self.new_length-2);
        2. This modification aims to allow dynamically sample more-than/less-than 10 clips;
"""

import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import torchvision
from transforms import *
import time
import math
import copy

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame{:06d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1 # Diff needs one more image to calculate diff
        self._parse_list()

    def _load_image(self, directory, idx):

        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    # def _sample_indices_BIT(self, record):
    #     """
    #     :param record: VideoRecord
    #     :return: list
    #     """
    #     import math

    #     starting = math.floor(record.num_frames * 0.2)
    #     ending = math.ceil(record.num_frames * 0.8)
    #     interval = ending - starting

    #     # average_duration = record.num_frames // self.new_length // self.num_segments
    #     average_duration = (interval) // self.new_length // self.num_segments
    #     segment_offset = average_duration * self.num_segments

    #     if average_duration > 0:
    #         # offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
    #         offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) + starting
    #     elif record.num_frames > self.num_segments:
    #         offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
    #     else:
    #         offsets = np.zeros((self.num_segments,))
    #     return (offsets + 1), int(segment_offset)

    def _sample_indices_BIT(self, record):

        """
        Implmentaton of sparse randomly sampling among evenly divided segments for training process;
        Keep same unit temporal interval between frames;

        Operation Flow:
            1. Remove pre-fix Staring and Ending;
            2. Split middle part to self.new_length parts;
            3. Randomly sample achors from clips, which is defined as self.new_length;
            4. Sample self.num_segments frames index from each achor continuously;

        Operation RGB:
            1. Remove pre-fix Starting and Ending;
            2. Split middle part to num_segments parts;
            3. Randomly sample one single frame from each segments;
            4. Next step, I should make sure the frame-wise interval is the same;
        """

        import math
        # starting = math.floor(record.num_frames * 0.15)
        # ending = math.ceil(record.num_frames * 0.85)
        starting = 0
        ending = record.num_frames - 1
        offsets = list()

        if self.modality == 'Flow':
            seg_range = np.linspace(starting, ending, self.new_length + 1)
            for i in range(self.new_length):
                anchor = np.random.randint(seg_range[i], seg_range[i+1] - self.num_segments, size=1)
                new_offset = list(range(anchor, anchor + self.num_segments))
                offsets.extend(new_offset)

        elif self.modality == 'RGB':
            seg_range = np.linspace(starting, ending, self.num_segments + 1)
            for i in range(self.num_segments):
                new_offset = np.random.randint(seg_range[i], seg_range[i+1], size=1)
                offsets.extend(new_offset)

        return (np.array(offsets) + 1)

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        """
            Assume sample 10 segments for testing; Each segment contains 5 frame index
            Operation:
                1. Find evenly divided index by 10 * 50; i.e. 89 / 50 = 1.78 ceilling to 2;
                2. Return first segment index as a = [0, num_segments];
                3. Compute the rest of segment index as b = [x+interval for x in a];
                4. For last segments (i==9), if out-range the total video length, then reservsely sample from [record.num_frames - 5, record.num_frames]
                5. Attention !!! for video that has longer sequence, there must be under-sample happending;
                6. Attention !!! for short term video, it is handled by over-lapping sampling, for now it should be fine;
        """
        if self.modality == 'Flow':
            interval = math.ceil(record.num_frames / self.new_length) # video length divded by 10;
            average_duration = (record.num_frames) // self.new_length
            # segment_offset = average_duration * self.num_segments

            # if average_duration > 0:
                # offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
                # offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
                # offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = range(self.num_segments) # First offset
            offsets_ref =copy.deepcopy(offsets)
            # print(offsets)

            if record.num_frames < (self.num_segments * self.new_length):
                for i in range(self.new_length-2):
                    new_offset = [x + (i+1) * interval for x in offsets_ref]
                    offsets.extend(new_offset)

                last_offset = range(record.num_frames-6, record.num_frames-1) # Remember it is not exclusive to lower end;
                offsets.extend(last_offset)
            else:
                ''' Find anchor and consecutively sampling frames '''
                for i in range(self.new_length-1):
                    new_offset = [x + (i+1) * interval for x in offsets_ref]
                    offsets.extend(new_offset)

                ''' uniformly sample [self.num_segs * self.new_length - 1] samples from overall record.num_frames '''
                # new_offset = np.linspace(0, record.num_frames-1, self.num_segments * self.new_length, dtype='int')
                # offsets = new_offset.astype('float64')
                # return (offsets + 1)

        elif self.modality == 'RGB':
            offsets = list()
            seg_range = np.linspace(0, record.num_frames, self.num_segments + 1)
            for i in range(self.num_segments):
                new_offset = np.random.randint(seg_range[i], seg_range[i+1], size=1)
                offsets.extend(new_offset)

        return (np.array(offsets) + 1)

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            # segment_indices, interval = self._sample_indices_BIT(record) if self.random_shift else self._get_val_indices(record)
            segment_indices = self._sample_indices_BIT(record) if self.random_shift else self._get_val_indices(record)
            return self.get(record, segment_indices)

        else:
            segment_indices = self._get_test_indices(record)
            # print(len(segment_indices))
            return self.get_test(record, segment_indices)

    def get(self, record, indices):

        # images = list()
        # for seg_ind in indices:
        #     p = int(seg_ind)
        #     # print(p)
        #     for i in range(self.new_length):
        #         # print(i, p)
        #         seg_imgs = self._load_image(record.path, p)

        #         images.extend(seg_imgs)
        #         if p < record.num_frames:
        #             p += interval

        # ''' Testing on sampled indices '''
        # # seg_2 = indices + interval
        # # seg_3 = seg_2 + interval

        # process_data = self.transform(images)
        # # return process_data, record.label, indices, seg_2, seg_3
        # return process_data, record.label

        # print(indices)
        images = list()
        for _, seg_ind in enumerate(indices):
            p = int(seg_ind)

            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record.label, indices


    def get_test(self, record, indices):
        """
            images list is arranged as [f1_seg1, f1_seg2, f1_seg3, f2_seg1, f2_seg2, f2_seg3, ...];

            So I only need to give indices that have same arrangement;
        """

        # images = list()
        # for idx, seg_ind in enumerate(indices):
        #     p = int(seg_ind)
        #     for i in range(self.new_length):
        #             seg_imgs = self._load_image(record.path, p)
        #             images.extend(seg_imgs)
        #             if p < record.num_frames:
        #                 p += (i+1) * interval
        # process_data = self.transform(images)
        # return process_data, record.label, indices

        images = list()
        for _, seg_ind in enumerate(indices):
            p = int(seg_ind)
            # print(p)

            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record.label, indices

    def __len__(self):
        return len(self.video_list)
