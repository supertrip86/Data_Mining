# -*- coding: utf-8 -*-
"""
Collections.deque()

"""

from collections import deque

d = deque()

n = int(input())

for i in range(n):
    cmd = input().split()
    
    if len(cmd) > 1:
        eval("d." + cmd[0] + "(" + cmd[1] + ")")
    else:
        eval("d." + cmd[0] + "()")

result = " ".join(list(map(str, d)))

print(result)
