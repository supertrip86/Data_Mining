# -*- coding: utf-8 -*-
"""
Company Logo

"""

from collections import Counter


if __name__ == '__main__':
    s = input()
    l = sorted([x for x in s])

    x = Counter(l).most_common()

    for i in range(3):
        print(" ".join(list(map(str, x[i]))))