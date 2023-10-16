# -*- coding: utf-8 -*-
"""
Zipped!

"""

a = list(map(int, input().split()))

N = a[0]
X = a[1]

data = []

for i in range(X):
    marks = list(map(float, input().split()))
    data.append(marks)

result = zip(*data)

for i in result:
    avg = sum(i) / X
    print("{:.1f}".format(avg))