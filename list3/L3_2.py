#!/usr/bin/env python3
# L3_2.py

import subprocess
subprocess.call(['pip', 'install', 'numpy'], stdout=subprocess.DEVNULL)

import time
import numpy as np
import random
import sys


def load_data():
    words = open('dict.txt', 'r').read().lower()
    words = words.split('\n')
    words = sorted(words)

    t, n, s = [int(el) for el in input().lower().split(' ') if len(el) > 0]
    letter_values = {}
    accepted_words = []

    for i in range(n):
        line = [el for el in input().lower().split(' ') if len(el) > 0]
        letter_values[line[0]] = int(line[1])

    for i in range(s):
        accepted_words.append([el for el in input().lower().split(' ') if len(el) > 0][0])

    possible_words = []

    for word in words:
        flag = True

        for letter in word:
            if letter not in letter_values.keys():
                flag = False

        if flag:
            possible_words.append(word)

    return t, n, s, letter_values, accepted_words, possible_words


def binary_search(words, word):
    lower = 0
    upper = len(words)
    while lower < upper:
        x = lower + (upper - lower) // 2
        val = words[x]
        if word == val:
            return True
        elif word > val:
            if lower == x:
                break
            lower = x
        elif word < val:
            upper = x

    return False


def run_search():
    t, n, s, letter_values, accepted_words, words = load_data()
    letters = list(letter_values.keys())

    def get_score(word):
        return sum([letter_values[char] for char in word])

    i = 0
    time_start = time.time()
    time_last = None

    while True:
        i += 1
        if i % 100000 == 0:
            accepted_words = [word for word in accepted_words if binary_search(words, word)]

        prob = np.random.random()

        if prob < 0.4:
            word = accepted_words[random.randint(0, len(accepted_words) - 1)]
            letter = letters[random.randint(0, len(letters) - 1)]

            index = random.randint(0, len(word) - 1)
            word = word[0:index] + letter + word[index:len(word)]
            accepted_words.append(word)
        elif prob < 0.8:
            word0 = accepted_words[random.randint(0, len(accepted_words) - 1)]
            word1 = accepted_words[random.randint(0, len(accepted_words) - 1)]
            accepted_words.append(word0 + word1)
            accepted_words.append(word1 + word0)
        elif prob < 1.0:
            word = []

            for _ in range(random.randint(1, 4)):
                word.append(letters[random.randint(0, len(letters) - 1)])

            word = ''.join(word)
            accepted_words.append(word)

        if not time_last is None:
            diff = time.time() - time_last
            pred_seconds = time.time() - time_start + diff
            if pred_seconds > t * 0.98:
                break

        time_last = time.time()

    accepted_words = [word for word in accepted_words if binary_search(words, word)]
    best_score = max([get_score(word) for word in accepted_words])
    best_words = [word for word in accepted_words if get_score(word) == best_score]

    return best_words[0], best_score


if __name__ == '__main__':
    best_word, best_score = run_search()
    print(best_score)
    print(best_word, file=sys.stderr)
