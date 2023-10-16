# -*- coding: utf-8 -*-
"""
The Captain's Room

"""

result = ''

d = {}

K = int(input())
el = input().split()

for i in el:
    if i in d:
        d[i] += 1
    else:
        d[i] = 1
        
for k in d:
    if d[k] == 1:
        result = int(k)

print(result)