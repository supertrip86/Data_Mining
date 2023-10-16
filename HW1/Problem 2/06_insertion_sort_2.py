# -*- coding: utf-8 -*-
"""
Insertion Sort - Part 2

"""

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(1, n):
        a = arr[i]

        for d in range( len(arr[0:i])):
            b = arr[d]

            if b > a:
                arr.insert(d, arr.pop(i))
                break

        print(" ".join(list(map(str, arr))))

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)