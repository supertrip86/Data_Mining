# -*- coding: utf-8 -*-
"""
Set Mutations

"""

n = int(input())
A = set(map(int, input().split()))

N = int(input())

for i in range(N):
    cmd = input().split()
    iSet = set(map(int, input().split()))
    
    eval("A." + cmd[0] + "(" + str(iSet) + ")")

print(sum(A))
