# -*- coding: utf-8 -*-
"""
Check Subset

"""

T = int(input())

for i in range(T):
    a = int(input())
    A = set(input().split())

    b = int(input())
    B = set(input().split())

    if A.issubset(B):
        print(True)
    else:
        print(False)