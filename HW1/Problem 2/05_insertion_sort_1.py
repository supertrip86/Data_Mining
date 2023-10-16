# -*- coding: utf-8 -*-
"""
Insertion Sort - Part 1

"""

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    value = arr[-1]

    for i in range(n-2, -1, -1):
        if arr[i] > value:
            arr[i+1] = arr[i]        
            print(" ".join( list(map(str, arr)) ))
        else:
            arr[i+1] = value
            print(" ".join( list(map(str, arr)) ))
            break
        if i == 0 and arr[0] > value:
            arr[0] = value
            print(" ".join( list(map(str, arr)) ))
        
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)