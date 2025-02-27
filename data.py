import os
import pickle

import numpy as np
import torch.utils.data
import tqdm
from torchvision.datasets import DatasetFolder

import myUtiles


def npy_loader(path):
    video, op_flow, bbox = np.load(path, allow_pickle=True)
    return torch.from_numpy(np.array(video)), torch.from_numpy(np.array(op_flow))


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=11):
        dataset = DatasetFolder(folder, npy_loader, extensions=('.npy',))
        self.total_frames = 0
        self.lengths = []
        self.arrays = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.arrays, self.lengths = pickle.load(f)
        else:
            for idx, (data, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames", leave=False)):
                array_path, _ = dataset.samples[idx]
                video, _ = data
                length = len(video)
                if length >= min_len:
                    self.arrays.append((array_path, categ))
                    self.lengths.append(length)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.arrays, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(("Total number of frames {}".format(np.sum(self.lengths))))

    def __getitem__(self, item):
        path, categ = self.arrays[item]
        video, op_flow, bbox = np.load(path, allow_pickle=True)
        return np.array(video), np.array(op_flow), np.array(bbox), categ

    def __len__(self):
        return len(self.arrays)


def general_transform(imgs):
    vid = []
    # channel_range = end_channel - start_channel
    # mean_tuple = tuple([0.5] * channel_range)
    # std_tuple = tuple([0.5] * channel_range)
    for img in imgs:
        img = torch.from_numpy(img.astype('float32')) / 255
        img = img.permute(2, 0, 1)
        # img = transforms.Normalize(mean_tuple, std_tuple)(img)
        vid.append(img)

    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth

    def __getitem__(self, item):
        video, op_flow, bbox, categ = self.dataset[item]
        if self.every_nth > 1:
            video = video[np.arange(0, stop=self.video_length, step=self.every_nth), :, :]
            op_flow = op_flow[np.arange(0, stop=self.video_length - 1, step=self.every_nth), :, :]
        video_input = video[:9, :, :]
        target_frames = video[9:11, :, :]
        target_frames = general_transform(target_frames)
        video_input = general_transform(video_input)

        bbox_input = bbox[:9, :, :]
        target_bboxs = bbox[9:11, :, :]
        target_bboxs = general_transform(target_bboxs)
        bbox_input = general_transform(bbox_input)

        input_flow = op_flow[:8, :, :]
        target_flow = op_flow[8:10, :, :]

        target_flow = general_transform(target_flow)
        input_flow = general_transform(input_flow)

        input_confidence = myUtiles.lable_crator(video_input, bbox_input)
        target_confidence = myUtiles.lable_crator(target_frames, target_bboxs)
        return video_input, input_flow, bbox_input, input_confidence, \
               target_frames, target_flow, target_bboxs, target_confidence

    def __len__(self):
        return len(self.dataset)
