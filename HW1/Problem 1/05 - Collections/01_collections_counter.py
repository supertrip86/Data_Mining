# -*- coding: utf-8 -*-
"""
collections.Counter()

"""

X = int(input())

x = input().split()

money = 0

N = int(input())

for i in range(N):
    a = input().split()
    
    if a[0] in x:
        money += int(a[1])
        x.remove(a[0])
        
print(money)