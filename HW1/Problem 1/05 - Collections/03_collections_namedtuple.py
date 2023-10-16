# -*- coding: utf-8 -*-
"""
Collections.namedtuple()

"""

N = int(input())
column = input().split().index('MARKS')

marks = []

for i in range(N):
    marks.append(int(input().split()[column]))
    
result = '{:.2f}'.format(sum(marks) / len(marks))

print(result)