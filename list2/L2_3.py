#!/usr/bin/env python3
# L2_3.py

import subprocess
subprocess.call(['pip3', 'install', 'numpy'], stdout=subprocess.DEVNULL)
import numpy as np
import sys


def load_data():
    seconds, n, m = [el for el in input().split(' ') if len(el) > 0]
    seconds, n, m = float(seconds), int(n), int(m)
    arr = np.zeros((n, m)).astype(int)

    for i in range(n):
        line = input().strip()
        for j in range(m):
            arr[i][j] = int(line[j])

    return arr, seconds


class AgentOnArray:
    def __init__(self, arr):
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.arr = arr
        self.agent_pos = None

        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                if arr[i, j] == 5:
                    self.agent_pos = (i, j)
                    self.arr[i, j] = 0

        self.agent_pos_history = [self.agent_pos]
        self.agent_move_history = []

    def get_str_arr(self):
        array = self.arr.tolist()

        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                if array[i][j] == 0:
                    array[i][j] = ' '
                elif array[i][j] == 1:
                    array[i][j] = '#'
                elif array[i][j] == 8:
                    array[i][j] = '.'

        for idx, el in enumerate(self.agent_pos_history):
            array[el[0]][el[1]] = str(idx%10)

        lines = [''.join(line) for line in array]
        str_array = '\n'.join(lines)
        print(self.agent_pos_history)

        return str_array

    def move(self, move_type):
        moves = self.moves.copy()
        increment = moves[move_type]
        check_position = (self.agent_pos[0]+increment[0], self.agent_pos[1]+increment[1])

        if self.arr[check_position[0], check_position[1]] in [0, 8]:
            self.agent_pos = check_position
            self.agent_pos_history.append(check_position)
            self.agent_move_history.append(move_type)
            return True

        return False

    def get_surrounding(self):
        moves = self.moves.copy()
        moves = {k: self.arr[self.agent_pos[0] + v[0], self.agent_pos[1] + v[1]] for (k, v) in moves.items()}
        return moves

    def perform_search(self):
        while True:
            surrounding = self.get_surrounding()

            if 8 in surrounding.values():
                end_dir = [k for (k, v) in surrounding.items() if v == 8][0]
                self.move(end_dir)
                break
            else:
                available_dirs = [k for (k, v) in surrounding.items() if v == 0]
                prob = {}

                for dir in available_dirs:
                    increment = self.moves[dir]
                    nex_pos = (self.agent_pos[0] + increment[0], self.agent_pos[1] + increment[1])
                    prob[dir] = len([pos for pos in self.agent_pos_history if pos == nex_pos]) + np.random.random()

                sum = np.sum([v for (k, v) in prob.items()])
                prob = {k: v / sum for (k, v) in prob.items()}

                dir = min(prob, key=prob.get)

                self.move(dir)

        return self.agent_move_history


if __name__ == "__main__":
    arr, seconds = load_data()
    agent = AgentOnArray(arr)
    move_history = agent.perform_search()
    print(len(move_history))
    print(' '.join(move_history), file=sys.stderr)
