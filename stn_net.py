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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        ngf = 128
        self.localization = nn.Sequential(
            nn.Conv2d(57, ngf, kernel_size=(3, 3), stride=1),
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

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(500, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, x_last):
        xs = self.localization(x)
        xs = xs.view(-1, 500)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x_last.size(), align_corners=True)
        x = F.grid_sample(x_last, grid, align_corners=True)

        return x

    def forward(self, x, x_flow):
        # transform the input
        x_last = x[:, :, -1, :, :]

        x = x.view(x.shape[0], -1, 256, 512)
        x_flow = x_flow.view(x_flow.shape[0], -1, 256, 512)
        x = torch.cat((x, x_flow), dim=1)

        x = self.stn(x, x_last)

        return x

        # TODO: chck if this is needed
        # # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


model = STN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0002)


def train(epoch, train_loader, writer):
    model.train()
    for batch_idx, (video_input, input_flow, target_frame, target_flow) in enumerate(
            tqdm(train_loader, leave=False, desc='train', ncols=100)):
        video_input, target_frame = video_input.to(device), target_frame.to(device)
        input_flow, target_flow = input_flow.to(device), target_flow.to(device)

        optimizer.zero_grad()
        output = model(video_input, input_flow)

        loss = F.mse_loss(output, target_frame) * 100
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            writer.add_scalar('Loss/train', loss.item(), batch_idx + epoch * len(train_loader))


#
# A simple test procedure to measure STN the performances on MNIST.
#


def test(epoch, test_loader, writer):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for video_input, input_flow, target_frame, target_flow in tqdm(test_loader, leave=False, desc='test',
                                                                       ncols=100):
            # transfer tensors to picked device
            video_input, target_frame = video_input.to(device), target_frame.to(device)
            input_flow, target_flow = input_flow.to(device), target_flow.to(device)

            output = model(video_input, input_flow)

            # sum up batch loss
            test_loss += F.mse_loss(output, target_frame).item() * 100

        test_loss /= len(test_loader)

        writer.add_scalar('Loss/test', test_loss, epoch)
        # Visualize the STN transformation on some input batch


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


def visualize_stn(epoch, test_loader, writer):
    with torch.no_grad():
        # Get a batch of training data
        video_input, input_flow, target_frame, target_flow = next(iter(test_loader))

        # transfer tensors back to cpu to prepare them to be shown
        video_input, target_frame = video_input.to(device), target_frame.cpu()
        input_flow, target_flow = input_flow.to(device), target_flow.cpu()

        output = model(video_input, input_flow).cpu()

        # in_grid = convert_image_np(
        #     torchvision.utils.make_grid(target_frame))
        #
        # out_grid = convert_image_np(
        #     torchvision.utils.make_grid(output))

        # # Plot the results side-by-side
        # # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(in_grid)
        # axarr[0].set_title('Target Images')

        N, C, H, W = output.shape

        fake_video = output.view(N, C, 1, H, W)
        fake_video = torch.cat((video_input.cpu(), fake_video), dim=2)
        fake_video = fake_video.permute(0, 2, 1, 3, 4)

        real_video = target_frame.view(N, C, 1, H, W)
        real_video = torch.cat((video_input.cpu(), real_video), dim=2)
        real_video = real_video.permute(0, 2, 1, 3, 4)

        writer.add_images('Image/Target_frame', target_frame, epoch)
        writer.add_images('Image/Fake_frame', output, epoch)
        writer.add_video('Video/Input_video_fake', fake_video, epoch, fps=2)
        writer.add_video('Video/Input_video_real', real_video, epoch, fps=2)

        # plt.imsave('real.png', in_grid)
        # plt.imsave('fake.png', out_grid)
        # plt.imsave('diff.png', abs(in_grid - out_grid))
        # axarr[1].imshow(out_grid)
        # axarr[1].set_title('Output Images')


train_folder = './data/train/'
test_folder = './data/test/'

if __name__ == '__main__':
    writer = SummaryWriter()

    # Training dataset
    train_dataset = data.VideoFolderDataset(train_folder, cache=os.path.join(train_folder, 'train.db'))
    train_video_dataset = data.VideoDataset(train_dataset, 11)
    train_loader = DataLoader(train_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)

    test_dataset = data.VideoFolderDataset(test_folder, cache=os.path.join(test_folder, 'test.db'))
    test_video_dataset = data.VideoDataset(test_dataset, 11)
    test_loader = DataLoader(test_video_dataset, batch_size=8, drop_last=True, num_workers=4, shuffle=True)

    for epoch in tqdm(range(0, 100), desc='epoch', ncols=100):
        train(epoch, train_loader, writer)
        test(epoch, test_loader, writer)
        visualize_stn(epoch, test_loader, writer)

    # to allow the tensorboard to flush the final data before the program close
    sleep(2)
