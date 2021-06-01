import glob

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

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


def Get_compare_video(video_input, bbox_input):
    filenames = glob.glob(".\\labels/*.png")
    filenames.sort()
    labels = [cv2.imread(img) for img in filenames]
    confidence_table = torch.zeros((video_input.shape[1], 37))
    for i in range(video_input.shape[1]):  # video length
        bbox = bbox_input[:, i, :, :].permute(1, 2, 0)
        x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
        crop_frame = video_input[:, i, x:x_w, y:y_h]
        imgs_to_compare = torch.zeros((37, 3, x_w - x, y_h - y))
        imgs_real = torch.zeros((37, 3, x_w - x, y_h - y))
        for j, label in enumerate(labels):
            interpolation = cv2.INTER_CUBIC if x_w - x > label.shape[1] else cv2.INTER_AREA
            label = cv2.resize(label, (y_h - y, x_w - x), interpolation=interpolation)
            label = torch.tensor(cv2.cvtColor(label, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            imgs_to_compare[j] = label/255
            imgs_real[j] = crop_frame
            # plt.imshow(label.permute(1, 2, 0) / 255)
            # plt.text = 'fake'
            # plt.show()
            # plt.imshow(crop_frame.permute(1, 2, 0))
            # plt.text = 'real'
            # plt.show()
        confidence_table[i] = SSIM.ssim(imgs_real, imgs_to_compare, size_average=False)
    return confidence_table
