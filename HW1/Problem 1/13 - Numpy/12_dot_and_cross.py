# -*- coding: utf-8 -*-
"""
Dot and Cross

"""

import numpy

x = list(map(int, input().split()))

a = []
b = []

for i in range(x[0]):
    y = list(map(int, input().split()))
    a.append(y)

for d in range(x[0]):
    z = list(map(int, input().split()))
    b.append(z)

result = numpy.matmul(a, b)

print(result)