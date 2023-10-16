# -*- coding: utf-8 -*-
"""
Matrix Script

"""

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

sep = ' '

string = ''

pattern = r'(?<=[A-Za-z0-9])([ !@#$%&]+)(?=[A-Za-z0-9])'

# Finds from one to unlimited characters (+ symbol) belonging to the set [ !@#$%&]
# that are preceded (?<=) and followed (?=) by one alphanumeric character [A-Za-z0-9].
# (?<=) is a positive lookbehind, while (?=) is a positive lookahead.

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

for d in range(m):
    for i in range(n):
        string += matrix[i][d]

m = re.sub(pattern, sep, string)

print(m)