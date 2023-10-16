# -*- coding: utf-8 -*-
"""
Linear Algebra

"""

import numpy

N = int(input())

A = []

for i in range(N):
    x = list(map(float, input().split()))
    A.append(x)

print(round(numpy.linalg.det(A), 2))