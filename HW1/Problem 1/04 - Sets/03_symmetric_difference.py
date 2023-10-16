# -*- coding: utf-8 -*-
"""
Symmetric Difference

"""

M = input()
value1 = input().split()
N = input()
value2 = input().split()

set1 = set(value1)
set2 = set(value2)

difference = list(set1.difference(set2)) + list(set2.difference(set1))
result = list(map(int, difference))
result.sort()

for i in result:
    print(i)
