# -*- coding: utf-8 -*-
"""
Viral Advertising

"""

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    values = [5]
    cumulative = 2

    if n > 1:
        for i in range(2, n + 1):
            shared = math.floor(values[i-2] // 2) * 3
            liked = shared // 2
            cumulative += liked
            values.append(shared)
            
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()