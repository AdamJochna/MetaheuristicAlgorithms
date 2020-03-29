#!/usr/bin/env python
# L1_1.py

import subprocess
subprocess.call(['pip', 'install', 'numpy'], stdout=subprocess.DEVNULL)

import numpy as np
import time
import sys


def happy_cat(x):
    result = np.power(np.abs(np.sum(np.power(x, 2), axis=1) - 4), 1/4)
    result += np.sum(np.power(x, 2), axis=1) / 8
    result += np.sum(x, axis=1) / 4
    result += 1/2
    return result


def griewank(x):
    result = np.ones(x.shape[0])
    result += np.sum(np.power(x, 2)/4000, axis=1)
    result -= np.prod(np.cos(x/(np.sqrt(np.arange(4)+1))), axis=1)
    return result


def particle_swarm_optimize(swarm_size, x_min, x_max, error_function, seconds):
    """
    each row of swarm array has swarm[i] =
    [
    0 - position_0,
    1 - position_1,
    2 - position_2,
    3 - position_3,
    4 - best_particle_position_0,
    5 - best_particle_position_1,
    6 - best_particle_position_2,
    7 - best_particle_position_3,
    8 - velocity_0,
    9 - velocity_1,
    10 - velocity_2,
    11 - velocity_3,
    12 - current_particle_error,
    13 - best_particle_error,
    ]
    """

    time_start = time.time()
    time_last = None

    swarm = np.empty((swarm_size, 14))
    swarm[:, 0:4] = np.random.random((swarm_size, 4)) * (x_max - x_min) + x_min  # initialize positions randomly
    swarm[:, 4:8] = swarm[:, 0:4]  # initialize best_positions with current positions
    swarm[:, 8:12] = np.random.random((swarm_size, 4)) * (x_max - x_min) + x_min  # initialize velocity randomly
    swarm[:, 12] = error_function(swarm[:, 0:4])  # initialize current positions error
    swarm[:, 13] = swarm[:, 12]  # initialize best error with current positions error

    best_swarm_idx = np.argmin(swarm[:, 13])
    best_swarm_pos = swarm[best_swarm_idx, 4:8]
    best_swarm_err = swarm[best_swarm_idx, 13]

    c0 = 0.729  # inertia
    c1 = 1.49445  # cognitive (particle)
    c2 = 1.49445  # social (swarm)

    while True:
        #  apply swarm update

        r1_vec = np.repeat(np.random.random(swarm_size)[:, np.newaxis], 4, 1)
        r2_vec = np.repeat(np.random.random(swarm_size)[:, np.newaxis], 4, 1)
        best_swarm_pos_arr = best_swarm_pos
        best_swarm_pos_arr = np.repeat(best_swarm_pos_arr[np.newaxis, :], swarm_size, 0)

        swarm[:, 8:12] = c0 * swarm[:, 8:12] + \
                         c1 * r1_vec * (swarm[:, 4:8] - swarm[:, 0:4]) + \
                         c2 * r2_vec * (best_swarm_pos_arr - swarm[:, 0:4])  # compute velocity

        swarm[:, 8:12] = np.where(swarm[:, 8:12] > x_max, x_max, swarm[:, 8:12])  # threshold too big velocity
        swarm[:, 8:12] = np.where(swarm[:, 8:12] < x_min, x_min, swarm[:, 8:12])  # threshold too small velocity

        swarm[:, 0:4] += swarm[:, 8:12]/100.0  # update position with velocity
        swarm[:, 12] = error_function(swarm[:, 0:4])  # update current error

        swarm[:, 4:8] = np.where(np.repeat(swarm[:, 13][:, np.newaxis], 4, 1) < np.repeat(swarm[:, 12][:, np.newaxis], 4, 1), swarm[:, 4:8], swarm[:, 0:4])
        swarm[:, 13] = np.where(swarm[:, 13] < swarm[:, 12], swarm[:, 13], swarm[:, 12])

        best_swarm_idx = np.argmin(swarm[:, 12])
        best_swarm_pos = swarm[best_swarm_idx, 0:4]
        best_swarm_err = swarm[best_swarm_idx, 12]

        if not time_last is None:
            diff = time.time() - time_last
            next_time = time.time() - time_start + diff
            if next_time > seconds*0.98:
                return best_swarm_pos, best_swarm_err
                break

        time_last = time.time()


if __name__ == "__main__":
    seconds, function_type = [el for el in input().split(' ') if len(el) > 0]
    seconds = float(seconds)

    if function_type == "0":
        best_swarm_pos, best_swarm_err = particle_swarm_optimize(2 ** 16, -2, 0, happy_cat, seconds)
    else:
        best_swarm_pos, best_swarm_err = particle_swarm_optimize(2 ** 16, -1, 1, griewank, seconds)

    print(' '.join([str(el) for el in best_swarm_pos] + [str(best_swarm_err)]))