# -*- coding: utf-8 -*-
"""
Concatenate

"""

import numpy

x = list(map(int, input().split()))

a = []
b = []

for i in range(x[0]):
    data = list(map(int, input().split()))
    a.append(data)

for d in range(x[1]):
    data = list(map(int, input().split()))
    b.append(data)

y = numpy.array(a)
z = numpy.array(b)

result = numpy.concatenate((y, z), axis = 0)

print(result)
