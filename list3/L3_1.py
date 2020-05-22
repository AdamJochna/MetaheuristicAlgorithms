#!/usr/bin/env python3
# L3_1.py

import subprocess
subprocess.call(['pip', 'install', 'numpy'], stdout=subprocess.DEVNULL)

import numpy as np
import time
import sys


def yang(x, eps):
    result = np.power(np.abs(x), np.arange(5)+1)
    result = np.dot(result, eps)
    return result


def particle_swarm_optimize(swarm_size, x_min, x_max, error_function, seconds, eps, x_init):
    """
        each row of swarm array has swarm[i] =
        [
        0 - position_0,
        1 - position_1,
        2 - position_2,
        3 - position_3,
        4 - position_4,
        5 - best_particle_position_0,
        6 - best_particle_position_1,
        7 - best_particle_position_2,
        8 - best_particle_position_3,
        9 - best_particle_position_4,
        10 - velocity_0,
        11 - velocity_1,
        12 - velocity_2,
        13 - velocity_3,
        14 - velocity_4,
        15 - current_particle_error,
        16 - best_particle_error,
        ]
    """

    time_start = time.time()
    time_last = None

    swarm = np.empty((swarm_size, 17))
    swarm[:, 0:5] = x_init  # initialize positions
    swarm[:, 5:10] = swarm[:, 0:5]  # initialize best_positions with current positions
    swarm[:, 10:15] = np.random.random((swarm_size, 5)) * (x_max - x_min) + x_min  # initialize velocity randomly
    swarm[:, 15] = error_function(swarm[:, 0:5], eps)  # initialize current positions error
    swarm[:, 16] = swarm[:, 15]  # initialize best error with current positions error

    best_swarm_idx = np.argmin(swarm[:, 16])
    best_swarm_pos = swarm[best_swarm_idx, 5:10]
    best_swarm_err = swarm[best_swarm_idx, 16]

    c0 = 0.729  # inertia
    c1 = 1.49445  # cognitive (particle)
    c2 = 1.49445  # social (swarm)

    while True:
        #  apply swarm update

        r1_vec = np.repeat(np.random.random(swarm_size)[:, np.newaxis], 5, 1)
        r2_vec = np.repeat(np.random.random(swarm_size)[:, np.newaxis], 5, 1)
        best_swarm_pos_arr = best_swarm_pos
        best_swarm_pos_arr = np.repeat(best_swarm_pos_arr[np.newaxis, :], swarm_size, 0)

        swarm[:, 10:15] = c0 * swarm[:, 10:15] + \
                         c1 * r1_vec * (swarm[:, 5:10] - swarm[:, 0:5]) + \
                         c2 * r2_vec * (best_swarm_pos_arr - swarm[:, 0:5])  # compute velocity

        swarm[:, 0:5] += swarm[:, 10:15]  # update position with velocity
        swarm[:, 0:5] = np.clip(swarm[:, 0:5], -5.0, 5.0)
        swarm[:, 15] = error_function(swarm[:, 0:5], eps)  # update current error

        swarm[:, 5:10] = np.where(np.repeat(swarm[:, 16][:, np.newaxis], 5, 1) < np.repeat(swarm[:, 15][:, np.newaxis], 5, 1), swarm[:, 5:10], swarm[:, 0:5])
        swarm[:, 16] = np.where(swarm[:, 16] < swarm[:, 15], swarm[:, 16], swarm[:, 15])

        best_swarm_idx = np.argmin(swarm[:, 15])
        best_swarm_pos = swarm[best_swarm_idx, 0:5]
        best_swarm_err = swarm[best_swarm_idx, 15]

        if not time_last is None:
            diff = time.time() - time_last
            next_time = time.time() - time_start + diff
            if next_time > seconds*0.98:
                return best_swarm_pos, best_swarm_err
                break

        time_last = time.time()


if __name__ == "__main__":
    input_list = [float(el) for el in input().split(' ') if len(el) > 0]
    t = input_list[0]
    x_init = np.array(input_list[1:6])
    eps = np.array(input_list[6:11])
    best_swarm_pos, best_swarm_err = particle_swarm_optimize(2 ** 16, -2, 0, yang, t, eps, x_init)

    print(' '.join([str(el) for el in best_swarm_pos] + [str(best_swarm_err)]))
