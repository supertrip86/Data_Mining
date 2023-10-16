# -*- coding: utf-8 -*-
"""
Set .intersection() Operation

"""

eng = int(input())
n = set(map(int, input().split()))

fre = int(input())
b = set(map(int, input().split()))

result = n.intersection(b)

print(len(result))

