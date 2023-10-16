# -*- coding: utf-8 -*-
"""
DefaultDict Tutorial

"""

from collections import defaultdict

d = defaultdict(list)

length = list(map(int, input().split()))
n = length[0]
m = length[1]

for i in range(n):
    d[input()].append(i + 1)

    
for c in range(m):
    a = input()
    if a in d:
        item = list(map(str, d[a]))
        result = " ".join(item)
    else:
        result = -1
    
    print(result)