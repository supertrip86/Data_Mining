# -*- coding: utf-8 -*-
"""
Set .union() Operation

"""

eng = int(input())
n = set(map(int, input().split()))

fre = int(input())
b = set(map(int, input().split()))

result = n.union(b)

print(len(result))
