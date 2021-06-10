# License: BSD
# Author: Ghassen Hamrouni

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        ngf = 128
        self.localization = nn.Sequential(
            nn.Conv2d(78, ngf, kernel_size=(3, 3), stride=1),
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

            # this was meant to be used for the scaled up pictures, now we use smaller pictures
            # nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(True),
            # nn.Conv2d(ngf * 8, ngf * 8, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(True),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(ngf * 4, 200, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(True),
            nn.Conv2d(200, 200, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(True),

            nn.AvgPool2d(kernel_size=(17, 32), stride=(17, 32))
        )

        # Regressors for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc2 = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, x_last, x_pred):
        xs = self.localization(x)
        xs = xs.view(-1, 200)
        theta1 = self.fc_loc1(xs[:, :100])
        theta1 = theta1.view(-1, 2, 3)
        theta2 = self.fc_loc2(xs[:, 100:200])
        theta2 = theta2.view(-1, 2, 3)

        grid1 = F.affine_grid(theta1, x_last.size(), align_corners=True)
        x_pred[:, :, 0, :, :] = F.grid_sample(x_last, grid1, align_corners=True)
        grid2 = F.affine_grid(theta2, x_last.size(), align_corners=True)
        x_pred[:, :, 1, :, :] = F.grid_sample(x_last, grid2, align_corners=True)

        return x_pred

    def forward(self, x, x_flow, x_pred, video_input):
        # transform the input
        x_last = x[:, :, -1, :, :]
        x = x.view(x.shape[0], -1, 200, 320)
        x_flow = x_flow.view(x_flow.shape[0], -1, 200, 320)
        video_input = video_input.view(video_input.shape[0], -1, 200, 320)
        x = torch.cat((x, video_input, x_flow), dim=1)

        x = self.stn(x, x_last, x_pred)

        return x
