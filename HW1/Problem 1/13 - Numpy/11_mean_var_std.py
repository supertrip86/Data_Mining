# -*- coding: utf-8 -*-
"""
Mean, Var, and Std

"""

import numpy

numpy.set_printoptions(legacy='1.13')

x = list(map(int, input().split()))

data = []

for i in range(x[0]):
    y = list(map(int, input().split()))
    data.append(y)

arr = numpy.array(data)

print(numpy.mean(arr, axis = 1))
print(numpy.var(arr, axis = 0))
print(numpy.std(arr, axis = None) )