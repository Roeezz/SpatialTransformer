import os
from time import sleep

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from myUtiles import Get_fake_video

import data
from lstm_net import LSTM
from stn_net import STN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_stn = STN().to(device)
opt_stn = optim.Adam(model_stn.parameters(), lr=0.00002)
model_lstm = LSTM().to(device)
opt_lstm = optim.Adam(model_lstm.parameters(), lr=0.0002)


def step_lstm(video_input, input_flow, input_labels, target_labels):
    label_vectors = torch.zeros((*input_labels.shape, 37)).to(device)
    for i in range(input_labels.shape[0]):
        for j in range(input_labels.shape[1]):
            label_vectors[i, j][int(input_labels[i, j])] = 1
    input_labels = label_vectors
    target_labels = target_labels.permute(1, 0)
    label_for_model = input_labels[:, 8, :]
    label_preds = torch.zeros((*target_labels.shape, 37)).to(device)
    output_labels = model_lstm(video_input, input_flow, label_for_model, label_preds)
    loss = F.nll_loss(output_labels.reshape(-1, 37), target_labels.reshape(-1))
    return loss


def step_stn(bbox_input, target_bboxs, input_flow, target_flow):
    video_pred = torch.zeros_like(target_bboxs).to(device)
    output = model_stn(bbox_input, input_flow, video_pred)
    loss = F.mse_loss(output, target_bboxs) * 100
    return loss


def train(epoch, train_loader, writer):
    model_lstm.train()
    model_stn.train()
    for batch_idx, (video_input, input_flow, bbox_input, input_labels,
                    target_frames, target_flow, target_bboxs, target_labels) in enumerate(
        tqdm(train_loader, leave=False, desc='train', ncols=100)):
        video_input, target_frames = video_input.to(device), target_frames.to(device)
        # bbox_input, target_bboxs = bbox_input.to(device), target_bboxs.to(device)
        input_flow, target_flow = input_flow.to(device), target_flow.to(device)
        input_labels, target_labels = input_labels.to(device), target_labels.to(device)
        opt_lstm.zero_grad()
        opt_stn.zero_grad()

        loss_lstm = step_lstm(video_input, input_flow, input_labels, target_labels)
        loss_lstm.backward()
        opt_lstm.step()

        loss_stn = step_stn(video_input, target_frames, input_flow, target_flow)
        loss_stn.backward()
        opt_stn.step()

        if batch_idx % 10 == 0:
            writer.add_scalar('Loss/train_lstm', loss_lstm.item(), batch_idx + epoch * len(train_loader))
            writer.add_scalar('Loss/train_stn', loss_stn.item(), batch_idx + epoch * len(train_loader))

PATH = './models/'
def test(epoch, test_loader, writer):
    with torch.no_grad():
        model_lstm.eval()
        model_stn.eval()
        test_loss_lstm = 0
        test_loss_stn = 0
        for batch_idx, (video_input, input_flow, bbox_input, input_confidence,
                        target_frames, target_flow, target_bboxs, target_confidence) in enumerate(
            tqdm(train_loader, leave=False, desc='test', ncols=100)):
            video_input, target_frames = video_input.to(device), target_frames.to(device)
            # bbox_input, target_bboxs = bbox_input.to(device), target_bboxs.to(device)
            input_flow, target_flow = input_flow.to(device), target_flow.to(device)
            input_confidence, target_confidence = input_confidence.to(device), target_confidence.to(device)

            test_loss_lstm += step_lstm(video_input, input_flow, input_confidence, target_confidence)
            test_loss_stn += step_stn(video_input, target_frames, input_flow, target_flow)
        loader_len = len(test_loader)
        test_loss_lstm /= loader_len
        test_loss_stn /= loader_len

        writer.add_scalar('Loss/test_lstm', test_loss_lstm, epoch)
        writer.add_scalar('Loss/test_stn', test_loss_stn, epoch)
        if epoch % 10:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_stn.state_dict(),
                'optimizer_state_dict': opt_stn.state_dict(),
                'loss': test_loss_stn,
            }, PATH)


# Visualize the STN transformation on some input batch
def visualize_stn(epoch, test_loader, writer):
    with torch.no_grad():
        # Get a batch of training data
        video_input, input_flow, bbox_input, input_confidence, \
        target_frames, target_flow, target_bboxs, target_confidence = next(iter(test_loader))

        # transfer tensors back to cpu to prepare them to be shown
        video_input, target_frame = video_input.to(device), target_frames.cpu()
        input_flow, target_flow = input_flow.to(device), target_flow.cpu()
        # bbox_input = bbox_input.to(device)
        target_confidence = target_confidence.permute(1, 0, 2)
        conf_for_model = input_confidence[:, 5, :].to(device)
        conf_pred = torch.zeros_like(target_confidence).to(device)
        video_pred = torch.zeros_like(target_frame).to(device)

        output_stn = model_stn(video_input, input_flow, video_pred).cpu()
        output_lstm = model_lstm(video_input, input_flow, conf_for_model, conf_pred).cpu()

        N, C, S, H, W = output_stn.shape
        fake_ending = Get_fake_video(output_lstm, output_stn)
        fake_video = torch.cat((video_input.cpu(), fake_ending), dim=2)
        fake_video = fake_video.permute(0, 2, 1, 3, 4)

        real_video = target_frames
        real_video = torch.cat((video_input.cpu(), real_video), dim=2)
        real_video = real_video.permute(0, 2, 1, 3, 4)

        writer.add_images('Image/Target_frame', target_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W), epoch)
        writer.add_images('Image/Fake_frame', fake_ending.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W), epoch)
        writer.add_video('Video/Input_video_fake', fake_video, epoch, fps=2)
        writer.add_video('Video/Input_video_real', real_video, epoch, fps=2)



train_folder = './data/train/'
test_folder = './data/test/'
if __name__ == '__main__':
    writer = SummaryWriter()

    # Training dataset
    train_dataset = data.VideoFolderDataset(train_folder, cache=os.path.join(train_folder, 'train.db'))
    train_video_dataset = data.VideoDataset(train_dataset, 11)
    train_loader = DataLoader(train_video_dataset, batch_size=6, drop_last=True, num_workers=3, shuffle=True)

    test_dataset = data.VideoFolderDataset(test_folder, cache=os.path.join(test_folder, 'test.db'))
    test_video_dataset = data.VideoDataset(test_dataset, 11)
    test_loader = DataLoader(test_video_dataset, batch_size=6, drop_last=True, num_workers=3, shuffle=True)

    for epoch in tqdm(range(0, 100000), desc='epoch', ncols=100):
        train(epoch, train_loader, writer)
        test(epoch, test_loader, writer)
        visualize_stn(epoch, test_loader, writer)

    # to allow the tensorboard to flush the final data before the program close

    sleep(2)
