# -*- coding: utf-8 -*-
"""
Arrays

"""

import numpy

def arrays(arr):
    newArr = list(map(float, arr))
    newArr.reverse()
    return numpy.array(newArr)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)