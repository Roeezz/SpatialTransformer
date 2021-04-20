import glob
import os
from time import sleep

import cv2
import numpy
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
labels = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(37, 37, num_layers=1)

        # Spatial transformer localization-network
        ngf = 128
        self.localization = nn.Sequential(
            nn.Conv2d(27, ngf, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf, ngf * 2, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 8, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),
            nn.Conv2d(500, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),

            nn.AvgPool2d(kernel_size=(7, 23), stride=(7, 23))
        )

        # convolution for the pictures
        self.input_encoder = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf, ngf * 2, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 8, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),
            nn.Conv2d(500, 500, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(500),
            nn.LeakyReLU(True),

            nn.AvgPool2d(kernel_size=(7, 23), stride=(7, 23))
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(500, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 37)
        )

        self.fc_enc = nn.Sequential(
            nn.Linear(500, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 37)
        )

    def forward(self, xs, xs_flow, bbox_targets):
        # transform the input
        batch_sz = bbox_targets.shape[0]
        seq_sz = bbox_targets.shape[2]
        bbox_targets = bbox_targets.permute(0, 2, 1, 3, 4)

        xs = xs.reshape(batch_sz, -1, 256, 512)
        xs_flow = xs_flow.reshape(batch_sz, -1, 256, 512)
        xs = torch.cat((xs, xs_flow), dim=1)
        bbox_targets = bbox_targets.reshape(batch_sz * seq_sz, -1, 256, 512)

        bbox_targets = self.input_encoder(bbox_targets)
        bbox_targets = bbox_targets.reshape(-1, 500)
        bbox_targets = self.fc_enc(bbox_targets)
        bbox_targets = bbox_targets.reshape(batch_sz, seq_sz, 37).permute(1, 0, 2)

        hidden = self.localization(xs)
        hidden = hidden.reshape(-1, 500)
        hidden = self.fc_loc(hidden).view(1, batch_sz, 37)
        out, _ = self.lstm(bbox_targets, (hidden, hidden))

        return out


model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0002)


def find_box_cords(a):
    a = numpy.array(a)
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        out = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
    else:
        out = (0, 0, 0, 0)
    return out


def train(epoch, train_loader, writer):
    model.train()
    for batch_idx, (video_input, input_flow, bbox_input, target_frame, target_flow, target_bbox) in enumerate(
            tqdm(train_loader, leave=False, desc='train', ncols=100)):
        video_target = video_input[:, :, 5:, :, :]
        video_input = video_input[:, :, :5, :, :]
        input_flow = input_flow[:, :, :4, :, :]
        bbox_input = bbox_input[:, :, 5:, :, :]
        video_input, target_frame = video_input.to(device), target_frame.to(device)
        bbox_input, target_bbox = bbox_input.to(device), target_bbox.to(device)
        input_flow, target_flow = input_flow.to(device), target_flow.to(device)
        optimizer.zero_grad()

        output = model(video_input, input_flow, bbox_input)
        get_loss(output, bbox_input, video_target)
    # loss.backward()
    # optimizer.step()


def get_loss(output, bbox_input, video_target):
    fake_vids = torch.zeros_like(video_target)
    for i in range(output.size(0)):  # batch size
        bboxs = bbox_input[i]
        out_batch = output[i]
        for j in range(bbox_input.shape[2]):  # seq size
            bbox = (bboxs[:, j, :, :])
            bbox = bbox.permute(1, 2, 0)
            x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
            interpolation = cv2.INTER_CUBIC if x_w >  else cv2.INTER_AREA
            imageB = cv2.resize(image, (imageA.shape[1], imageA.shape[0]), interpolation=interpolation)
            fake_vids[i, :, j, x:x_w, y:y_h] = [:, x: x_w, y: y_h]

    train_folder = './data/train/'
    test_folder = './data/test/'

    if __name__ == '__main__':
        # Training dataset
        train_dataset = data.VideoFolderDataset(train_folder, cache=os.path.join(train_folder, 'train.db'))
        train_video_dataset = data.VideoDataset(train_dataset, 11)
        train_loader = DataLoader(train_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)

        filenames = glob.glob("labels/*.png")
        filenames.sort()
        labels.extend([cv2.imread(img) for img in filenames])

        for epoch in tqdm(range(0, 100), desc='epoch', ncols=100):
            train(epoch, train_loader, None)

        # to allow the tensorboard to flush the final data before the program close
        sleep(2)
