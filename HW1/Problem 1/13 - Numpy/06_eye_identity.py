# -*- coding: utf-8 -*-
"""
Eye and Identity

"""

import numpy

numpy.set_printoptions(sign=' ')

x = list(map(int, input().split()))

print(numpy.eye(x[0], x[1]))
