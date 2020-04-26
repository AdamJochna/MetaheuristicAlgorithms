#!/usr/bin/env python3
# L2_1.py

import subprocess
subprocess.call(['pip3', 'install', 'numpy'], stdout=subprocess.DEVNULL)
import numpy as np
import time


def cost_function(x):
    tmp = np.sqrt(np.sum(np.power(x, 2)))
    result = 1.0 - np.cos(2.0 * np.pi * tmp) + 0.1 * tmp
    return result


def random_neighbour(x, frac):
    delta = (np.random.random(4) - 1/2)*2
    x = x + delta
    x = np.clip(x, a_min=-5, a_max=5)
    return x


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature) / 10
        return p


def make_annealing_step(x, cost, step, steps):
    frac = step / float(steps)
    T = max(0.01, min(1.0, 1.0 - frac))
    new_x = random_neighbour(x, frac)
    new_cost = cost_function(new_x)

    if acceptance_probability(cost, new_cost, T) > np.random.random():
        return new_x, new_cost
    else:
        return x, cost


def annealing(seconds, x):
    cost = cost_function(x)

    time_start = time.time()
    time_last = None
    pred_seconds = 0

    while True:
        for i in range(100):
            x, cost = make_annealing_step(x, cost, pred_seconds, seconds)

        if not time_last is None:
            diff = time.time() - time_last
            pred_seconds = time.time() - time_start + diff
            if pred_seconds > seconds*0.98:
                return x, cost
                break

        time_last = time.time()

    return x, cost_function(x)


if __name__ == "__main__":
    input_list = [el for el in input().split(' ') if len(el) > 0]
    seconds = float(input_list[0])
    x_init = np.array([float(x) for x in input_list[1:]])
    x, cost = annealing(seconds, x_init)
    print(' '.join([str(el) for el in x.tolist() + [cost]]))
