#!/usr/bin/env python3
# L2_2.py

import subprocess
subprocess.call(['pip3', 'install', 'numpy'], stdout=subprocess.DEVNULL)
import numpy as np
import time
import sys
import random


def load_data():
    t, n, m, k, = [int(el) for el in input().split(' ') if len(el) > 0]
    M = np.zeros((n, m), dtype=np.int)

    for i in range(n):
        line = [el for el in input().split(' ') if len(el) > 0]
        for j in range(m):
            M[i][j] = int(line[j])

    return t, n, m, k, M


def mse_arrays(M, Mp):
    return np.mean(np.power(M-Mp, 2))


def mse_of_locs(M, loc_list):
    values = np.array([0, 32, 64, 128, 160, 192, 223, 255], dtype=np.int)
    idxs = 0
    sum = 0

    for loc in loc_list:
        M_part = M[loc[0]:loc[1], loc[2]:loc[3]]
        idxs += (loc[1] - loc[0])*(loc[3] - loc[2])
        avg = np.mean(M_part)
        best_value = values[np.argmin(np.abs(values-avg))]
        sum += np.sum(np.power(M_part-best_value, 2))

    return sum/idxs


def approx_matrix_from_locs(M, locs):
    Mp = np.empty(M.shape, dtype=np.int)
    values = np.array([0, 32, 64, 128, 160, 192, 223, 255], dtype=np.int)

    for loc in locs:
        M_part = M[loc[0]:loc[1], loc[2]:loc[3]]
        avg = np.mean(M_part)
        best_value = values[np.argmin(np.abs(values - avg))]
        Mp[loc[0]:loc[1], loc[2]:loc[3]] = best_value
        Mp[loc[0]:loc[1], loc[2]:loc[3]] = best_value

    return Mp


def get_best_block_array(t, n, m, k, M):
    locs = [(0, n, 0, m)]
    ready_locs = []

    while len(locs) > 0:
        n0, n1, m0, m1 = locs.pop()
        splits = []

        for i in range(n0 + k, n1 - k + 1):
            loc1, loc2 = (n0, i, m0, m1), (i, n1, m0, m1)
            splits.append([loc1, loc2])

        for j in range(m0 + k, m1 - k + 1):
            loc1, loc2 = (n0, n1, m0, j), (n0, n1, j, m1)
            splits.append([loc1, loc2])

        splits = [[loc1, loc2, mse_of_locs(M, [loc1, loc2])] for loc1, loc2 in splits]

        if len(splits) > 0:
            if np.random.random() < 0.5:
                loc1, loc2, mse = random.choice(splits)
            else:
                loc1, loc2, mse = min(splits, key=lambda x: x[2])

            locs.append(loc1)
            locs.append(loc2)
        else:
            ready_locs.append((n0, n1, m0, m1))

    return approx_matrix_from_locs(M, ready_locs)


def get_best_array(t, n, m, k, M):
    best_mse = np.inf
    best_arr = np.zeros((n, m))

    time_start = time.time()
    time_last = None

    while True:
        Mp = get_best_block_array(t, n, m, k, M)

        if mse_arrays(M, Mp) < best_mse:
            best_mse = mse_arrays(M, Mp)
            best_arr = Mp

        if not time_last is None:
            diff = time.time() - time_last
            pred_seconds = time.time() - time_start + diff
            if pred_seconds > t * 0.98:
                break

        time_last = time.time()

    return best_mse, best_arr


if __name__ == "__main__":
    t, n, m, k, M = load_data()
    best_mse, best_arr = get_best_array(t, n, m, k, M)
    print(best_mse)

    for i in range(n):
        print(' '.join([str(best_arr[i, j]) for j in range(m)]), file=sys.stderr)
