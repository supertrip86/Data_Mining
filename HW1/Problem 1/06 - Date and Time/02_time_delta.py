# -*- coding: utf-8 -*-
"""
Time Delta

"""

import datetime as dt
import math
import os
import random
import re
import sys

# Complete the time_delta function below.
def time_delta(t1, t2):
    x = dt.datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    y = dt.datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')

    result = (x-y).total_seconds()

    return str(round(abs(result)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()
