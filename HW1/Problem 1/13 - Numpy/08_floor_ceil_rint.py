# -*- coding: utf-8 -*-
"""
Floor, Ceil and Rint

"""

import numpy

numpy.set_printoptions(sign=' ')

x = list(map(float, input().split()))

arr = numpy.array(x)

print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))
