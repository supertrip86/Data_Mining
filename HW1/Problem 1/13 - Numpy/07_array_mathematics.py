# -*- coding: utf-8 -*-
"""
Array Mathematics

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

A = numpy.array(a)
B = numpy.array(b)

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)
