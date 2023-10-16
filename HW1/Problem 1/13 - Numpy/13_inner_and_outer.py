# -*- coding: utf-8 -*-
"""
Inner and Outer

"""

import numpy

data = [list(map(int, input().split())), list(map(int, input().split()))]

A = numpy.array(data[0])
B = numpy.array(data[1])

print(numpy.inner(A, B))
print(numpy.outer(A, B))