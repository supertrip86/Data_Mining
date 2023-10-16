# -*- coding: utf-8 -*-
"""
Min and Max

"""

import numpy

x = list(map(int, input().split()))

data = []

for i in range(x[0]):
    y = list(map(int, input().split()))
    data.append(y)

arr = numpy.array(data)

a = numpy.min(arr, axis = 1)
b = numpy.max(a)

print(b)