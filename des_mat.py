import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
def get_matrix():
    groups = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 56), (56, 64), (64, 69), (69, 74), (74, 82)]
    group_move_metrix = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                                  [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                                  [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                                  [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  ])

    data_file = open('label_data/data.pkl', 'rb')
    data = pickle.load(data_file)
    data_file.close()
    mat = np.zeros((82, 82))
    print(mat.shape)
    prev_key = None
    prev_val = None

    for i, (key, val) in enumerate(data.items()):
        if prev_key and prev_val:
            mat[prev_val, val] += 1
            prev_key = key
            prev_val = val
        elif i % 11 == 0:
            prev_key = key
            prev_val = val

    mat += np.identity(mat.shape[0])

    for i in range(group_move_metrix.shape[0]):
        for j in range(group_move_metrix.shape[1]):
            if group_move_metrix[i, j] == 0:
                mat[groups[i][0]:groups[i][1], groups[j][0]:groups[j][1]] *= 0
            elif j >= 8 and i < 8:
                mat[groups[i][0]:groups[i][1], groups[j][0]:groups[j][1]] *= 0.3
            elif j == i:
                mat[groups[i][0]:groups[i][1], groups[j][0]:groups[j][1]] *= 1.5

    for g in groups:
        for j in range(g[0],g[1]-1):
            mat[j,j+1]+=1

    for i in range(mat.shape[0]):
        if np.sum(mat[i]) > 0:
            mat[i] = mat[i] / np.sum(mat[i])
    # plt.matshow(mat, cmap='hot')
    # plt.show()
    return mat

if __name__ == '__main__':
    plt.matshow(get_matrix(), cmap='hot')
    plt.show()