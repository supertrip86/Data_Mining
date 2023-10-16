# -*- coding: utf-8 -*-
"""
Set .discard(), .remove() & .pop()

"""

n = int(input())
s = set(map(int, input().split()))

N = int(input())

for i in range(N):
    inp = input().split()
    cmd = inp[0]
    par = "()" if (cmd == "pop") else "(" + inp[1] + ")"

    eval("s." + cmd + par)
    
print(sum(s))