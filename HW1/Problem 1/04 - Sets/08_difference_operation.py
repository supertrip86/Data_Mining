# -*- coding: utf-8 -*-
"""
Set .difference() Operation

"""

eng = int(input())
n = set(map(int, input().split()))

fre = int(input())
b = set(map(int, input().split()))

result = n.difference(b)

print(len(result))