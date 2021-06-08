import glob
import re

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image as im

import SSIM


def find_box_cords(a):
    a = np.array(a)
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        out = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
    else:
        out = (0, 0, 0, 0)
    return out


def Get_fake_video(lstm_output, stn_output):
    fake_vids = torch.zeros_like(stn_output)
    for i in range(lstm_output.size(1)):  # batch size
        bboxs = stn_output[i]
        out_batch = lstm_output[:, i, :]
        for j in range(stn_output.shape[2]):  # seq size
            bbox = (bboxs[:, j, :, :])
            bbox = bbox.permute(1, 2, 0)
            x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
            label = torch.argmax(out_batch[j]).data
            label_img = cv2.imread('./labels/' + str(int(label)) + '.png')
            interpolation = cv2.INTER_CUBIC if x_w - x > label_img.shape[1] else cv2.INTER_AREA
            label_img = cv2.resize(label_img, (y_h - y, x_w - x), interpolation=interpolation)
            label_img = torch.tensor(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            fake_vids[i, :, j, x:x_w, y:y_h] = label_img
            # plt.imshow(fake_vids[i, :, j, x:x_w, y:y_h].permute(1, 2, 0) / 255)
            # plt.text = 'fake'
            # plt.show()
            # plt.imshow(video_target[i, :, j, x:x_w, y:y_h].permute(1, 2, 0))
            # plt.text = 'real'
            # plt.show()
    return fake_vids / 255


def take_num_from_file(item):
    num = re.search('\d+', item).group()
    return int(num)


def Get_compare_video(video_input, bbox_input):
    filenames = glob.glob(".\\labels/*.png")
    filenames = sorted(filenames, key=take_num_from_file)
    labels = [cv2.imread(img) for img in filenames]
    confidence_table = torch.zeros((video_input.shape[1], 37))
    confidence_table_ratio = torch.zeros((video_input.shape[1], 37))
    for i in range(video_input.shape[1]):  # video length
        bbox = bbox_input[:, i, :, :].permute(1, 2, 0)
        x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
        crop_frame = video_input[:, i, x:x_w, y:y_h]
        imgs_to_compare = torch.zeros((37, 3, x_w - x, y_h - y))
        imgs_real = torch.zeros((37, 3, x_w - x, y_h - y))
        for j, label in enumerate(labels):
            frame_ratio = crop_frame.shape[1] / crop_frame.shape[2]
            label_ratio = label.shape[0] / label.shape[1]
            confidence_table_ratio[i, j] = 1 if abs(frame_ratio - label_ratio) < 0.5 else -1
            interpolation = cv2.INTER_LANCZOS4 if x_w - x > label.shape[1] else cv2.INTER_NEAREST
            label = cv2.resize(label, (y_h - y, x_w - x), interpolation=interpolation)
            label = torch.tensor(cv2.cvtColor(label, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            imgs_to_compare[j] = label / 255
            imgs_real[j] = crop_frame
            # plt.imshow(label.permute(1, 2, 0) / 255)
            # plt.title(f'label is {str(j)}')
            # plt.show()
            # plt.imshow(crop_frame.permute(1, 2, 0))
            # plt.text = 'real'
            # plt.show()
        confidence_table[i] = SSIM.ssim(imgs_real, imgs_to_compare, size_average=False)
    return torch.minimum(confidence_table, confidence_table_ratio)


def Get_compare_video2(video_input, bbox_input):
    filenames = glob.glob(".\\labels/*.png")
    filenames = sorted(filenames, key=take_num_from_file)
    labels = [cv2.imread(img) for img in filenames]
    confidence_table = torch.zeros((video_input.shape[1], 37))
    for i in range(video_input.shape[1]):  # video length
        frame = video_input[:, i, :, :]
        for j, label in enumerate(labels):
            frame2 = frame.permute(1, 2, 0).numpy()
            img1 = label
            img2 = frame2
            img2 = (img2 * 255).astype(np.uint8)

            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            count = 0
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    count += 1

            confidence_table[i, j] = len(matchesMask)
    return confidence_table
