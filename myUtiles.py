import glob
import re
from time import sleep

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
    filenames = glob.glob("labels/*.png")
    filenames = sorted(filenames, key=take_num_from_file)
    labels = [cv2.imread(img) for img in filenames]
    confidence_table = torch.zeros((video_input.shape[1], 82))
    confidence_table_ratio = torch.zeros((video_input.shape[1], 82))
    for i in range(video_input.shape[1]):  # video length
        bbox = bbox_input[:, i, :, :].permute(1, 2, 0)
        x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
        crop_frame = video_input[:, i, x:x_w, y:y_h]
        imgs_to_compare = torch.zeros((82, 3, x_w - x, y_h - y))
        imgs_real = torch.zeros((82, 3, x_w - x, y_h - y))
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


groups = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 56), (56, 64), (64, 69), (69, 74), (74, 82)]


def group_conf(confidence_table):
    group_cofidence = torch.zeros((11))
    best_lable = torch.zeros((confidence_table.shape[0]))
    for j in range(confidence_table.shape[0]):
        group_cofidence[0] = torch.mean(confidence_table[j][groups[0][0]:groups[0][1]])
        group_cofidence[1] = torch.mean(confidence_table[j][groups[1][0]:groups[1][1]])
        group_cofidence[2] = torch.mean(confidence_table[j][groups[2][0]:groups[2][1]])
        group_cofidence[3] = torch.mean(confidence_table[j][groups[3][0]:groups[3][1]])
        group_cofidence[4] = torch.mean(confidence_table[j][groups[4][0]:groups[4][1]]) -1
        group_cofidence[5] = torch.mean(confidence_table[j][groups[5][0]:groups[5][1]])
        group_cofidence[6] = torch.mean(confidence_table[j][groups[6][0]:groups[6][1]])
        group_cofidence[7] = torch.mean(confidence_table[j][groups[7][0]:groups[7][1]])
        group_cofidence[8] = torch.mean(confidence_table[j][groups[8][0]:groups[8][1]])
        group_cofidence[9] = torch.mean(confidence_table[j][groups[9][0]:groups[9][1]])
        group_cofidence[10] = torch.mean(confidence_table[j][groups[10][0]:groups[10][1]]) -1
        sorted_groups = torch.argsort(group_cofidence, descending=True)
        for k, group_arg in enumerate(sorted_groups):
            confidence_table[j][groups[group_arg][0]:groups[group_arg][1]] += 0
        best_lable[j] = torch.argmax(confidence_table[j])

    return best_lable


def remove_background(img):
    img_ret = img
    alpha = np.array(img[:, :, 3]) / 255
    img_ret = cv2.cvtColor(img_ret, cv2.COLOR_BGRA2BGR)
    img_ret[:, :, 0] = np.array(img_ret[:, :, 0]) * alpha
    img_ret[:, :, 1] = np.array(img_ret[:, :, 1]) * alpha
    img_ret[:, :, 2] = np.array(img_ret[:, :, 2]) * alpha
    img_ret = cv2.cvtColor(img_ret, cv2.COLOR_BGR2GRAY)
    return img_ret


def Get_compare_video2(video_input, bbox_input):
    filenames = glob.glob("labels/*.png")
    filenames = sorted(filenames, key=take_num_from_file)
    labels = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in filenames]
    labels = [remove_background(img) for img in labels]
    confidence_table = torch.zeros((video_input.shape[1], 82))
    confidence_table_ratio = torch.zeros((video_input.shape[1], 82))
    sift = cv2.SIFT_create()
    for i in range(video_input.shape[1]):  # video length
        # frame = video_input[:, i, :, :]
        # frame2 = frame.permute(1, 2, 0).numpy()
        # img2 = frame2
        # img2 = (img2 * 255).astype(np.uint8)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        bbox = bbox_input[:, i, :, :].permute(1, 2, 0)
        x, x_w, y, y_h = find_box_cords(bbox[:, :, 0])
        crop_frame = video_input[:, i, x:x_w, y:y_h]
        crop_frame = (crop_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_ratio = crop_frame.shape[1] / crop_frame.shape[2]
        img2 = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2GRAY)

        keypoints_2, descriptors_2 = sift.detectAndCompute(crop_frame, None)
        time = 2
        while not keypoints_2:
            keypoints_2, descriptors_2 = sift.detectAndCompute(crop_frame * time, None)
            time += 2
        for j, label in enumerate(labels):
            label_ratio = label.shape[0] / label.shape[1]
            confidence_table_ratio[i, j] = 999999

            img1 = label

            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
            # Apply ratio test
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
            confidence_table[i, j] = len(good)

            # img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3), plt.show()

    return torch.minimum(confidence_table, confidence_table_ratio)


def lable_crator(video_input, bboxs):
    x = Get_compare_video2(video_input, bboxs)
    return group_conf(x)
# def check_homography(mat):
#     # check 1:Compute the determinant of the homography, and see if it's too close
#     #      to zero for comfort
#     if not mat.empty():
#         mat_det = np.linalg.det(mat)
#         if mat_det > 0.1:
#             return True
#         else:
#             return False
#     else:
#         return False
#
#     #
#     /* Check 2. Compute its SVD, and verify that the ratio of the first-to-last
#      singular value is not too high (order of 1.0E7). */
#     Mat singularValues = new Mat();
#     Core.SVDecomp(homography_mat, singularValues, new Mat(), new Mat(), Core.SVD_NO_UV);
#
#     System.out.print("\n Printing the singular values of the homography");
#     for (int i = 0; i < singularValues.rows(); i++){
#         for ( int j = 0; j < singularValues.cols(); j++){
#             System.out.print("\n Element at ( " + i + ", " + j + " ) is " + singularValues.get(i, j)[0]);
#         }
#     }
#     double conditionNumber = singularValues.get(0, 0)[0] / singularValues.get(2, 0)[0];
#     System.out.print("\n Condition number is : " + conditionNumber);
#
#     if(conditionNumber < Math.pow(10, 7)){
#         System.out.print("\n Homography matrix is non-singular");
#         return true;
#         }
#     else{
#         System.out.print("\n Homography matrix is singular (or very close)");
#         return false;
#         }
#     /* Check 3. Check the compare absolute values at (0,0) and (0,1) with (1,1) and (1,0)
#      * respectively. If the two differences are close to 0, the homography matrix is
#      * good. (This just takes of rotation and not translation)
#      * */
#     if(Math.abs((Math.abs(homography_mat.get(0, 0)[0]) - Math.abs(homography_mat.get(1, 1)[0]))) <= 0.1){
#         if(Math.abs((Math.abs(homography_mat.get(0, 1)[0]) - Math.abs(homography_mat.get(1, 0)[0]))) <= 0.1){
#             System.out.print("\n The homography matrix is good");
#             return true;
#         }
#     }
#         else{
#             System.out.print("\n The homography matrix is bad");
#             return false;
#         }
#     return false;
#     /*
#      * Check 4: If the determinant of the top-left 2 by 2 matrix (rotation) > 0, transformation is orientation
#      * preserving.
#      * Else if the determinant is < 0, it is orientation reversing
#      *
#      * */
#      Determinant of the rotation mat
#     double det = homography_mat.get(0, 0)[0] * homography_mat.get(1,1)[0] - homography_mat.get(0, 1)[0] * homography_mat.get(1, 0)[0];
#     if (det < 0)
#         return false;
#
#     double N1 = Math.sqrt(homography_mat.get(0, 0)[0] * homography_mat.get(0, 0)[0] + homography_mat.get(1, 0)[0] * homography_mat.get(1, 0)[0]);
#     if (N1 > 4 || N1 < 0.1)
#         return false;
#
#     double N2 = Math.sqrt(homography_mat.get(0, 1)[0] * homography_mat.get(0, 1)[0] + homography_mat.get(1, 1)[0] * homography_mat.get(1, 1)[0]);
#     if (N2 > 4 || N2 < 0.1)
#         return false;
#
#     double N3 = Math.sqrt(homography_mat.get(2, 0)[0] * homography_mat.get(2, 0)[0] + homography_mat.get(2,1)[0] * homography_mat.get(2, 1)[0]);
#     if (N3 < 0.002)
#         return false;
#
#     return true;
#
# }
