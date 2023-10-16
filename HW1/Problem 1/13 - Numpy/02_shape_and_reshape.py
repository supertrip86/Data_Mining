# -*- coding: utf-8 -*-
"""
Shape and Reshape

"""

import numpy

x = list(map(int, input().split()))

arr = numpy.array(x)

result = numpy.reshape(arr, (3,3))

print(result)