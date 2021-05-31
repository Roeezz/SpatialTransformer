import os
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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

    def forward(self, xs, xs_flow, bbox_targets, conf_for_model):
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
        out = conf_for_model
        hidden_in = (hidden, hidden)
        output = torch.zeros((bbox_targets.shape[0], batch_sz, 37))
        for i in range(bbox_targets.shape[0]):
            out, hidden_in = self.lstm(out.view(1, batch_sz, 37), hidden_in)
            output[i] = out

        return output


model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0002)


def train(epoch, train_loader, writer):
    model.train()
    for batch_idx, (video_input, input_flow, bbox_input, input_confidence,
                    target_frame, target_flow, target_bbox, target_confidence) in enumerate(
        tqdm(train_loader, leave=False, desc='train', ncols=100)):
        # video_target = video_input[:, :, 5:, :, :]
        video_input = video_input[:, :, :5, :, :]
        input_flow = input_flow[:, :, :4, :, :]
        bbox_input = bbox_input[:, :, 5:, :, :]
        confidence_target = input_confidence[:, 5:, :].permute(1, 0, 2)
        conf_for_model = input_confidence[:, 4, :]
        video_input, target_frame = video_input.to(device), target_frame.to(device)
        bbox_input, target_bbox = bbox_input.to(device), target_bbox.to(device)
        input_flow, target_flow = input_flow.to(device), target_flow.to(device)
        input_confidence, confidence_target = input_confidence.to(device), confidence_target.to(device)
        optimizer.zero_grad()

        output_confidence = model(video_input, input_flow, bbox_input, conf_for_model)
        # TODO: optionally use KLDIV loss instead
        # output_confidence = F.log_softmax(output_confidence, dim=2)
        # confidence_target = F.log_softmax(confidence_target, dim=2)
        # loss = F.kl_div(output_confidence, confidence_target, reduction='batchmean', log_target=True)
        loss = F.l1_loss(output_confidence, confidence_target)

        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            writer.add_scalar('Loss/train', loss.item(), batch_idx + epoch * len(train_loader))


def test(epoch, test_loader, writer):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for batch_idx, (video_input, input_flow, bbox_input, input_confidence,
                        target_frame, target_flow, target_bbox, target_confidence) in enumerate(
            tqdm(train_loader, leave=False, desc='test', ncols=100)):
            # transfer tensors to picked device
            # video_target = video_input[:, :, 5:, :, :]
            video_input = video_input[:, :, :5, :, :]
            input_flow = input_flow[:, :, :4, :, :]
            bbox_input = bbox_input[:, :, 5:, :, :]
            confidence_target = input_confidence[:, 5:, :].permute(1, 0, 2)
            # input_confidence = input_confidence[:, :4, :].permute(1, 0, 2)
            conf_for_model = input_confidence[:, 4, :]
            video_input, target_frame = video_input.to(device), target_frame.to(device)
            bbox_input, target_bbox = bbox_input.to(device), target_bbox.to(device)
            input_flow, target_flow = input_flow.to(device), target_flow.to(device)
            input_confidence, confidence_target = input_confidence.to(device), confidence_target.to(device)

            output_confidence = model(video_input, input_flow, bbox_input, conf_for_model)
            # TODO: optionally use KLDIV loss instead
            # output_confidence = F.log_softmax(output_confidence, dim=2)
            # confidence_target = F.log_softmax(confidence_target, dim=2)
            # test_loss += F.kl_div(output_confidence, confidence_target, reduction='batchmean', log_target=True)
            test_loss += F.l1_loss(output_confidence, confidence_target)
        test_loss /= len(test_loader)

        writer.add_scalar('Loss/test', test_loss, epoch)
        # Visualize the STN transformation on some input batch


train_folder = './data/train/'
test_folder = './data/test/'

if __name__ == '__main__':
    writer = SummaryWriter()
    # Training dataset
    train_dataset = data.VideoFolderDataset(train_folder, cache=os.path.join(train_folder, 'train.db'))
    train_video_dataset = data.VideoDataset(train_dataset, 11)
    train_loader = DataLoader(train_video_dataset, batch_size=2, drop_last=True, num_workers=1, shuffle=True)

    test_dataset = data.VideoFolderDataset(test_folder, cache=os.path.join(test_folder, 'test.db'))
    test_video_dataset = data.VideoDataset(test_dataset, 11)
    test_loader = DataLoader(test_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)

    for epoch in tqdm(range(0, 100), desc='epoch', ncols=100):
        train(epoch, train_loader, writer)
        test(epoch, test_loader, writer)

    # to allow the tensorboard to flush the final data before the program close
    sleep(2)
