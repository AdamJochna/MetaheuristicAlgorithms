#!/usr/bin/env python
# L1_2.py

import subprocess
subprocess.call(['pip', 'install', 'numpy'], stdout=subprocess.DEVNULL)

import numpy as np
import time
import sys


def load_data():
    seconds, n = [el for el in input().split(' ') if len(el) > 0]
    seconds, n = float(seconds), int(n)
    graph = np.zeros((n, n))

    for i in range(n):
        line = [el for el in input().split(' ') if len(el) > 0]
        for j in range(n):
            graph[i][j] = int(line[j])

    return graph, seconds


def get_greedy_cycle(graph, first_vertice):
    graph = graph.copy()
    n = graph.shape[0]
    cycle = []

    for i in range(0, n):
        graph[i, i] = np.inf

    vertice = first_vertice
    graph[:, vertice] = np.inf
    cycle.append(vertice)

    for _ in range(1, n):
        vertice = np.argmin(graph[vertice, :])
        graph[:, vertice] = np.inf
        cycle.append(vertice)

    cycle.append(first_vertice)

    return cycle


def cycle_cost(graph, cycle):
    costs = [graph[cycle[i], cycle[i+1]] for i in range(len(cycle)-1)]
    return sum(costs)


def update_tabu_dict(swap, tabu_dict, tabu_expiration_time):
    tabu_dict = {k: v-1 for (k, v) in tabu_dict.items() if v-1 > 0}

    if not (swap is None):
        tabu_dict[swap] = tabu_expiration_time

    return tabu_dict


def perform_swap(graph, cycle, tabu_dict, tabu_expiration_time):
    graph = graph.copy()
    swaps = []

    for i in range(1, len(cycle)-1):
        for j in range(i + 1, len(cycle)-1):
            change_set = list(set([i - 1, i, j - 1, j]))
            cycle_swapped = cycle.copy()
            cycle_swapped[i], cycle_swapped[j] = cycle_swapped[j], cycle_swapped[i]
            change = 0

            for idx in change_set:
                change -= graph[cycle[idx], cycle[idx + 1]]
                change += graph[cycle_swapped[idx], cycle_swapped[idx + 1]]

            swaps.append([i, j, cycle_swapped[i], cycle_swapped[j], change])

    swaps = sorted(swaps, key=lambda swap: swap[4])
    non_tabu_found = False

    for swap in swaps:
        if not (swap[2], swap[3]) in tabu_dict.keys():
            cycle[swap[0]], cycle[swap[1]] = cycle[swap[1]], cycle[swap[0]]
            update_tabu_dict((swap[2], swap[3]), tabu_dict, tabu_expiration_time)
            non_tabu_found = True
            break

    if not non_tabu_found:
        update_tabu_dict(None, tabu_dict, tabu_expiration_time)


def optimize_tabu_tsp(graph, seconds):
    time_start, time_last = time.time(), None
    best_cost, best_cycle = np.inf, None
    tabu_dict = {}

    for i in range(graph.shape[0]):
        cycle = get_greedy_cycle(graph, i)
        cost = cycle_cost(graph, cycle)

        if cost < best_cost:
            best_cost, best_cycle = cost, cycle

    cycle = best_cycle

    while True:
        tabu_randomized_expiration = int(graph.shape[0]/10 + 10*np.random.random())
        perform_swap(graph, cycle, tabu_dict, tabu_randomized_expiration)
        cost = cycle_cost(graph, cycle)

        if cost < best_cost:
            best_cost, best_cycle = cost, cycle

        if not (time_last is None):
            diff = time.time() - time_last
            next_time = time.time() - time_start + diff

            if next_time > seconds*0.98:
                return best_cycle, best_cost
                break

        time_last = time.time()


if __name__ == "__main__":
    graph, seconds = load_data()
    best_cycle, best_cost = optimize_tabu_tsp(graph, seconds)
    print(best_cost)
    print(' '.join([str(el) for el in best_cycle]), file=sys.stderr)