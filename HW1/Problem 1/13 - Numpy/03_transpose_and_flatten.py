# -*- coding: utf-8 -*-
"""
Transpose and Flatten

"""

import numpy

x = list(map(int, input().split()))

a = x[0]
b = x[1]

data = []

for i in range(a):
    y = list(map(int, input().split()))
    data.append(y)

arr = numpy.array([z for z in data])

transpose = numpy.transpose(arr)

flatten = arr.flatten()

print(transpose)
print(flatten)
