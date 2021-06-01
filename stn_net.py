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
            nn.Conv2d(51, ngf, kernel_size=(3, 3), stride=1),
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
            nn.Linear(250, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc2 = nn.Sequential(
            nn.Linear(250, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        # self.fc_loc3 = nn.Sequential(
        #     nn.Linear(500, 32),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(32, 3 * 2)
        # )
        #
        # self.fc_loc4 = nn.Sequential(
        #     nn.Linear(500, 32),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(32, 3 * 2)
        # )
        #
        # self.fc_loc5 = nn.Sequential(
        #     nn.Linear(500, 32),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(32, 3 * 2)
        # )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # self.fc_loc3[2].weight.data.zero_()
        # self.fc_loc3[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        #
        # self.fc_loc4[2].weight.data.zero_()
        # self.fc_loc4[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        #
        # self.fc_loc5[2].weight.data.zero_()
        # self.fc_loc5[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, x_last, x_pred):
        xs = self.localization(x)
        xs = xs.view(-1, 500)
        theta1 = self.fc_loc1(xs[:, :250])
        theta1 = theta1.view(-1, 2, 3)
        theta2 = self.fc_loc2(xs[:, 250:500])
        theta2 = theta2.view(-1, 2, 3)
        # theta3 = self.fc_loc3(xs)
        # theta3 = theta3.view(-1, 2, 3)
        # theta4 = self.fc_loc4(xs)
        # theta4 = theta4.view(-1, 2, 3)
        # theta5 = self.fc_loc5(xs)
        # theta5 = theta5.view(-1, 2, 3)

        grid1 = F.affine_grid(theta1, x_last.size(), align_corners=True)
        x_pred[:, :, 0, :, :] = F.grid_sample(x_last, grid1, align_corners=True)
        grid2 = F.affine_grid(theta2, x_last.size(), align_corners=True)
        x_pred[:, :, 1, :, :] = F.grid_sample(x_last, grid2, align_corners=True)
        # grid3 = F.affine_grid(theta3, x_last.size(), align_corners=True)
        # x_pred[:, :, 2, :, :] = F.grid_sample(x_last, grid3, align_corners=True)
        # grid4 = F.affine_grid(theta4, x_last.size(), align_corners=True)
        # x_pred[:, :, 3, :, :] = F.grid_sample(x_last, grid4, align_corners=True)
        # grid5 = F.affine_grid(theta5, x_last.size(), align_corners=True)
        # x_pred[:, :, 4, :, :] = F.grid_sample(x_last, grid5, align_corners=True)

        return x_pred

    def forward(self, x, x_flow, x_pred):
        # transform the input
        x_last = x[:, :, -1, :, :]
        x = x.view(x.shape[0], -1, 256, 512)
        x_flow = x_flow.view(x_flow.shape[0], -1, 256, 512)
        x = torch.cat((x, x_flow), dim=1)

        x = self.stn(x, x_last, x_pred)

        return x
