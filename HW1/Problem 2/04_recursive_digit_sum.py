# -*- coding: utf-8 -*-
"""
Recursive Digit Sum

"""

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    a = sum(list(map(int, [i for i in str(n)])))
    digitSum = a * k

    if digitSum > 9:
        return superDigit(digitSum, 1)

    return digitSum

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()