# -*- coding: utf-8 -*-
"""
Collections.OrderedDict()

"""

from collections import OrderedDict

od = OrderedDict()

N = int(input())

for i in range(N):
    data = input().split()
    price = int(data[-1])
    data.pop()
    item = " ".join(data)
    if item in od:
        od[item] += price
    else:
        od[item] = price

for c in od:
    print(c + ' ' + str(od[c]))