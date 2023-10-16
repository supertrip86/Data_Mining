# -*- coding: utf-8 -*-
"""
Set .symmetric_difference() Operation

"""

eng = int(input())
n = set(map(int, input().split()))

fre = int(input())
b = set(map(int, input().split()))

result = n.symmetric_difference(b)

print(len(result))