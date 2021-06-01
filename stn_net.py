# License: BSD
# Author: Ghassen Hamrouni

import os
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data

# for matplotlib.pyplot debugging with plt.imshow
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        ngf = 128
        self.localization = nn.Sequential(
            nn.Conv2d(33, ngf, kernel_size=(3, 3), stride=1),
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

        # Regressors for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc2 = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc3 = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc4 = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc5 = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc3[2].weight.data.zero_()
        self.fc_loc3[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc4[2].weight.data.zero_()
        self.fc_loc4[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc5[2].weight.data.zero_()
        self.fc_loc5[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, x_last, x_pred):
        xs = self.localization(x)
        xs = xs.view(-1, 500)
        theta1 = self.fc_loc1(xs)
        theta1 = theta1.view(-1, 2, 3)
        theta2 = self.fc_loc2(xs)
        theta2 = theta2.view(-1, 2, 3)
        theta3 = self.fc_loc3(xs)
        theta3 = theta3.view(-1, 2, 3)
        theta4 = self.fc_loc4(xs)
        theta4 = theta4.view(-1, 2, 3)
        theta5 = self.fc_loc5(xs)
        theta5 = theta5.view(-1, 2, 3)

        grid1 = F.affine_grid(theta1, x_last.size(), align_corners=True)
        x_pred[:, :, 0, :, :] = F.grid_sample(x_last, grid1, align_corners=True)
        grid2 = F.affine_grid(theta2, x_last.size(), align_corners=True)
        x_pred[:, :, 1, :, :] = F.grid_sample(x_last, grid2, align_corners=True)
        grid3 = F.affine_grid(theta3, x_last.size(), align_corners=True)
        x_pred[:, :, 2, :, :] = F.grid_sample(x_last, grid3, align_corners=True)
        grid4 = F.affine_grid(theta4, x_last.size(), align_corners=True)
        x_pred[:, :, 3, :, :] = F.grid_sample(x_last, grid4, align_corners=True)
        grid5 = F.affine_grid(theta5, x_last.size(), align_corners=True)
        x_pred[:, :, 4, :, :] = F.grid_sample(x_last, grid5, align_corners=True)

        return x_pred

    def forward(self, x, x_flow, x_pred):
        # transform the input
        x_last = x[:, :, -1, :, :]
        x = x.view(x.shape[0], -1, 256, 512)
        x_flow = x_flow.view(x_flow.shape[0], -1, 256, 512)
        x = torch.cat((x, x_flow), dim=1)

        x = self.stn(x, x_last, x_pred)

        return x


# model = STN().to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=0.0002)


# def train(epoch, train_loader, writer):
#     model.train()
#     for batch_idx, (video_input, input_flow, bbox_input, target_frame, target_flow, target_bbox) in enumerate(
#             tqdm(train_loader, leave=False, desc='train', ncols=100)):
#         video_input, target_frame = video_input.to(device), target_frame.to(device)
#         input_flow, target_flow = input_flow.to(device), target_flow.to(device)
#         optimizer.zero_grad()
#         output = model(video_input, input_flow)
#
#         loss = F.mse_loss(output, target_frame) * 100
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 5 == 0:
#             writer.add_scalar('Loss/train', loss.item(), batch_idx + epoch * len(train_loader))


#
# A simple test procedure to measure STN the performances on MNIST.
#


# def test(epoch, test_loader, writer):
#     with torch.no_grad():
#         model.eval()
#         test_loss = 0
#         for video_input, input_flow, target_frame, target_flow in tqdm(test_loader, leave=False, desc='test',
#                                                                        ncols=100):
#             # transfer tensors to picked device
#             video_input, target_frame = video_input.to(device), target_frame.to(device)
#             input_flow, target_flow = input_flow.to(device), target_flow.to(device)
#
#             output = model(video_input, input_flow)
#
#             # sum up batch loss
#             test_loss += F.mse_loss(output, target_frame).item() * 100
#
#         test_loss /= len(test_loader)
#
#         writer.add_scalar('Loss/test', test_loss, epoch)
#         # Visualize the STN transformation on some input batch


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.



# train_folder = './data/train/'
# test_folder = './data/test/'

# if __name__ == '__main__':
#     writer = SummaryWriter()
#
#     # Training dataset
#     train_dataset = data.VideoFolderDataset(train_folder, cache=os.path.join(train_folder, 'train.db'))
#     train_video_dataset = data.VideoDataset(train_dataset, 11)
#     train_loader = DataLoader(train_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)
#
#     test_dataset = data.VideoFolderDataset(test_folder, cache=os.path.join(test_folder, 'test.db'))
#     test_video_dataset = data.VideoDataset(test_dataset, 11)
#     test_loader = DataLoader(test_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)
#
#     for epoch in tqdm(range(0, 100), desc='epoch', ncols=100):
#         train(epoch, train_loader, writer)
#         test(epoch, test_loader, writer)
#         visualize_stn(epoch, test_loader, writer)
#
#     # to allow the tensorboard to flush the final data before the program close
#     sleep(2)
