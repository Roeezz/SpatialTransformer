import os
import pickle

import numpy as np
import torch.utils.data
import tqdm
from torchvision.datasets import DatasetFolder


def npy_loader(path):
    return torch.from_numpy(np.load(path))


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
            for idx, (array, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                array_path, _ = dataset.samples[idx]
                length = len(array)
                if length >= min_len:
                    self.arrays.append((array_path, categ))
                    self.lengths.append(length)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.arrays, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(("Total number of frames {}".format(np.sum(self.lengths))))

    def __getitem__(self, item):
        path, label = self.arrays[item]
        arr = np.load(path) / 255
        return arr, label

    def __len__(self):
        return len(self.arrays)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]

        frame = video[frame_num]
        if frame.shape[0] == 0:
            print(("video {}. num {}".format(video.shape, item)))

        return {"images": self.transforms(frame), "categories": target}

    def __len__(self):
        return self.dataset.cumsum[-1]


def general_transform(imgs, start_channel=0, end_channel=3):
    vid = []
    # channel_range = end_channel - start_channel
    # mean_tuple = tuple([0.5] * channel_range)
    # std_tuple = tuple([0.5] * channel_range)
    for img in imgs:
        img = torch.from_numpy(img)
        img = img[:, :, start_channel:end_channel]
        img = img.permute(2, 0, 1)
        # img = transforms.Normalize(mean_tuple, std_tuple)(img)
        vid.append(img)

    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, t_in, t_out, every_nth=1):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth

    def __getitem__(self, item):
        video_and_flow, target = self.dataset[item]

        video_len = video_and_flow.shape[0]
        # videos can be of various length, we randomly sample sub-sequences
        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            raise Exception("Frame skip is too high id - {}, len - {}, frame skip - {}").format(self.dataset[item],
                                                                                                video_len,
                                                                                                self.every_nth)
        else:
            raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)

        selected = video_and_flow[subsequence_idx]
        optical_flow = calc_optical_flow(selected[:, :, :, 0:3])
        optical_flow = optical_flow.astype('float32')
        optical_flow = general_transform(optical_flow)
        images = general_transform(selected/255)
        depth = general_transform(selected, 6, 7)
        mask = np.zeros_like(images)
        mask[images > 0] = 1.0
        mask = (mask[0, ...] + mask[1, ...] + mask[2, ...]) / 3.0
        mask = mask.reshape((1, *mask.shape))
        return {"images": images[:, :self.t_in, ...], "depth": depth[:, :self.t_in, ...],
                "optical_flow": optical_flow[:, :self.t_in - 1, ...], "input_mask": mask[:, :self.t_in, ...],
                "label_images": images[:, self.t_in:self.t_out, ...], "label_mask": mask[:, self.t_in:self.t_out, ...]}

    def __len__(self):
        return len(self.dataset)
