# -*- coding: utf-8 -*-
"""
Re.start() & Re.end()

"""

import re

string = input()
substring = input()

h = r'' + substring + ''

i = 0

result = []

def iterSearch(h, string, i):
    m = re.search(h, string)

    if m:
        result.append(tuple([m.start() + i, m.end() + i -1]))
        i += m.start() + 1
        iterSearch(h, string[m.start() + 1:], i)

iterSearch(h, string, i)

if len(result):
    for i in result:
        print(i)
else:
    print(tuple([-1, -1]))