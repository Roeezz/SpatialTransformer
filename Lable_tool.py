import glob
import os
import pickle
import cv2
import numpy as np
import torch
import random
import myUtiles
from data import general_transform


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


command_list = [('help', 'print the list of commands'), ('n', 'give next ssim guess'),
                ('y', 'set label as the ssim guess'), ('a or enter', 'set the current as the last set label')
    , ('r', 'save a random guess'), ('{num}', 'set label as the num given'),
                ('done/exit', 'save the data to file and exit')]
# the command list is
if __name__ == '__main__':

    filenames = glob.glob(".\\labels/*.png")
    filenames = sorted(filenames, key=myUtiles.take_num_from_file)
    labels = [cv2.imread(img) for img in filenames]
    data_folder = 'data/train/TROOP/'
    last_guess = 0
    data_file = open('label_data/data.pkl', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    for filename in os.listdir(data_folder):
        if filename.endswith('.npy'):
            video, _, bboxs = np.load(data_folder + filename, allow_pickle=True)
            video = general_transform(video)
            bboxs = general_transform(bboxs)
            confidence_values = myUtiles.Get_compare_video(video, bboxs)
            video = video.permute(1, 2, 3, 0)
            bboxs = bboxs.permute(1, 2, 3, 0)
            confidence = torch.argsort(confidence_values, dim=1)
            for img_num, (img, bbox, conf, conf_val) in enumerate(zip(video, bboxs, confidence, confidence_values)):
                if f'{filename}_{str(img_num)}' in data:
                    continue
                img = img.permute(2, 0, 1)
                x, x_w, y, y_h = myUtiles.find_box_cords(bbox[:, :, 0])
                img = img[:, x:x_w, y:y_h]
                guss_index = 0
                x = 'n'
                img = img.permute(1, 2, 0)
                while x == 'n':
                    print(img.shape)
                    print(labels[conf[guss_index]].shape)
                    print(conf_val[conf[guss_index]])
                    cv2.imshow('label', cv2.resize(labels[conf[guss_index]], (labels[conf[guss_index]].shape[1]*8, labels[conf[guss_index]].shape[0]*8)))
                    cv2.imshow('image', cv2.resize(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR), (img.shape[1]*8, img.shape[0]*8)))
                    cv2.waitKey(1)
                    x = input(f'{filename}_{str(img_num)} label guess is {conf[guss_index]}: ')
                    if x == 'n' and guss_index < 37:
                        guss_index += 1
                    elif x == 'y':
                        last_guess = conf[guss_index]
                        data[f'{filename}_{str(img_num)}'] = conf[guss_index]
                        print(conf[guss_index])
                    elif x == 'a' or x == '':
                        data[f'{filename}_{str(img_num)}'] = last_guess
                        print(last_guess)
                    elif x == 'r':
                        r = random.randint(0, 37)
                        data[f'{filename}_{str(img_num)}'] = r
                        print(r)
                    elif RepresentsInt(x):
                        last_guess = int(x)
                        data[f'{filename}_{str(img_num)}'] = int(x)
                        print(int(x))
                    elif x == 'done' or x == 'exit':
                        print('save and exit')
                        data_file = open('label_data/data.pkl', 'wb')
                        pickle.dump(data, data_file)
                        data_file.close()
                        exit(0)
                    elif x == 'help' or x == 'h':
                        x = 'n'
                        for c in command_list:
                            print(c)
                    elif x[0] == 's' and RepresentsInt(x[1:]):
                        label_to_see = int(x[1:])
                        cv2.imshow('show_img', cv2.resize(labels[label_to_see], (480, 480)))
                        cv2.waitKey(1)
                        while x != 'n' and x != 'y':
                            x = input('is it you label y/n: ')
                        if x == 'y':
                            data[f'{filename}_{str(img_num)}'] = label_to_see
                            last_guess = label_to_see
                        cv2.destroyWindow('show_img')
                    else:
                        x = 'n'
                        print('type help if you need some')
