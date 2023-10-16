# -*- coding: utf-8 -*-
"""
Piling Up!

"""

from collections import deque

T = int(input())

for d in range(T):
    n = int(input())
    sideLengths = deque(list(map(int, input().split())))

    result = "Yes"

    while len(sideLengths) > 2:
        left = sideLengths[0]
        right = sideLengths[-1]
        if left >= right and left >= sideLengths[1]:
            sideLengths.popleft()
        elif right >= left and right >= sideLengths[-2]:
            sideLengths.pop()
        else:
            result = "No"
            break

    print(result)