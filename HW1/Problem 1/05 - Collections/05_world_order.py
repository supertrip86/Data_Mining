# -*- coding: utf-8 -*-
"""
Word Order

"""

from collections import defaultdict

n = int(input())

d = defaultdict(list)

for i in range(n):
    word = input()
    if word in d:
        d[word] += 1
    else:
        d[word] = 1
        
occurrences = list(map(str, [d[x] for x in d]))
print(str(len(d)))
print(" ".join(occurrences))