# -*- coding: utf-8 -*-
"""
Zeros and Ones

"""

import numpy

x = tuple(map(int, input().split()))

print(numpy.zeros(x, dtype = numpy.int))
print(numpy.ones(x, dtype = numpy.int))
