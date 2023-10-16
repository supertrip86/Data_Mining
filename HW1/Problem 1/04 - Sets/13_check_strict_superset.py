# -*- coding: utf-8 -*-
"""
Check Strict Superset

"""

A = set(input().split())

N = int(input())

result = False

for i in range(N):
    X = set(input().split())

    if A.issuperset(X):
        result = True
    else:
        result = False
        break

print(result)