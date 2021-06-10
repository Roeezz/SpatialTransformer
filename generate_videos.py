import os
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import data
from des_mat import get_matrix
from myUtiles import Get_fake_video
from myUtiles import get_next_two_labels
from stn_net import STN

model_path = "./model_stn0000240"
out_path = "./fake_videos"
model_stn = STN().cpu()
des_mat = get_matrix()


def calc_optical_flow(video):
    """k
    :param video: video
    :type video:np.array
    :return: optical flow as an array of len(video)-1
    :rtype: np.array
    """
    img_array_new = np.zeros_like(video, dtype=np.uint8)
    size = (video.shape[1], video.shape[2])
    for i in range(len(video)):
        new_image = np.zeros((video[i].shape[0], video[i].shape[1], 3), dtype=np.uint8)
        new_image[:, :, 0] = video[i][:, :, 2]
        new_image[:, :, 1] = video[i][:, :, 1]
        new_image[:, :, 2] = video[i][:, :, 0]
        img_array_new[i] = new_image

    first_frame = img_array_new[0]

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255
    out_optical = np.zeros((len(img_array_new) - 1, *size, 3), dtype=np.uint8)
    for i in range(1, len(img_array_new)):
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        frame = img_array_new[i]
        # Opens a new window and displays the input
        # frame
        # cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # from shape (channel , width , height) to (width , height , channel)
        # rgb = rgb.transpose((2, 1, 0))
        # Flip image to back to rgb
        flip_rgb = np.zeros_like(rgb, dtype=np.uint8)
        flip_rgb[:, :, 0] = rgb[:, :, 2]
        flip_rgb[:, :, 1] = rgb[:, :, 1]
        flip_rgb[:, :, 2] = rgb[:, :, 0]

        out_optical[i - 1] = flip_rgb

        # Updates previous frame
        prev_gray = gray
    return out_optical


def generate_video(test_loader):
    checkpoint = torch.load(model_path)
    model_stn.load_state_dict(checkpoint['model_state_dict'])
    model_stn.eval()
    with torch.no_grad():
        # Get a batch of training data
        video_input, input_flow, bbox_input, input_labels, \
        target_frames, target_flow, target_bboxs, target_labels = next(iter(test_loader))
        # transfer tensors back to cpu to prepare them to be shown
        video_input, target_frame = video_input.cpu(), target_frames.cpu()
        input_flow, target_flow = input_flow.cpu(), target_flow.cpu()
        bbox_input = bbox_input.cpu()
        label_for_model = input_labels[:, 8].cpu()
        video_pred = torch.zeros_like(target_frame).cpu()
        new_videos = video_input
        for i in range(30):
            output_stn = model_stn(bbox_input, input_flow, video_pred, video_input).cpu()

            N, C, S, H, W = output_stn.shape
            next_labels = get_next_two_labels(label_for_model, des_mat)
            label_for_model = next_labels[:, 1]
            fake_ending = Get_fake_video(next_labels, output_stn)
            new_videos = torch.cat((new_videos, fake_ending), dim=2)
            video_input = torch.cat((video_input, fake_ending), dim=2)
            bbox_input = torch.cat((bbox_input, output_stn), dim=2)
            video_input = video_input[:, :, 2:, :, :]
            bbox_input = bbox_input[:, :, 2:, :, :]
            for k, vid in enumerate(video_input):
                input_flow[k] = torch.tensor(calc_optical_flow(vid.permute(1, 2, 3, 0).numpy())).permute(3, 0, 1, 2)
            print(f'finished creating {2 * (i + 1)} frames')

        fps = 35
        new_videos = new_videos.permute(0, 2, 3, 4, 1)
        new_videos = new_videos.numpy()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i, vid in enumerate(new_videos):
            out = cv2.VideoWriter(f'{out_path}/vid_{i}.avi', cv2.VideoWriter_fourcc(*"MPEG"), fps, (320, 200))
            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame *= 255
                frame = frame.astype(np.uint8)
                out.write(frame)
                cv2.imshow("frame", frame)
                cv2.waitKey(100)
            out.release()
            sleep(2)


test_folder = './data/test/'
if __name__ == '__main__':
    test_dataset = data.VideoFolderDataset(test_folder, cache=os.path.join(test_folder, 'test.db'))
    test_video_dataset = data.VideoDataset(test_dataset, 11)
    test_loader = DataLoader(test_video_dataset, batch_size=5, drop_last=True, num_workers=3, shuffle=True)

    generate_video(test_loader)
