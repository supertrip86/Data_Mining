# -*- coding: utf-8 -*-
"""
No Idea!

"""

length = input()
array = input().split()
A = set(input().split())
B = set(input().split())

happiness = 0

union = A.union(B)

newArr = [x for x in array if (x in union) ]

for element in newArr:
    if element in A:
        happiness += 1
    elif element in B:
        happiness -= 1

print(happiness)
