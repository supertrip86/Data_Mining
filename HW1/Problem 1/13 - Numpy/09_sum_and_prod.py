# -*- coding: utf-8 -*-
"""
Sum and Prod

"""

import numpy

x = list(map(int, input().split()))

data = []

for i in range(x[0]):
    y = list(map(int, input().split()))
    data.append(y)

arr = numpy.array(data)

a = numpy.sum(arr, axis = 0)
b = numpy.prod(a)

print(b)