# -*- coding: utf-8 -*-
"""
Alphabet Rangoli

"""

d = {i:chr(i+96) for i in range(1,27)}

def print_rangoli(size):
    w = ((size - 1) * 4 ) + 1
    values = []
    for i in range(size, 0, -1):
        h = values[:]
        h.reverse()
        s = values + [d[i]] + h
        print("-".join(s).center(w, "-"))
        values.append(d[i])
    values.pop()
    for i in range(2, size + 1):
        values.pop()
        h = values[:]
        h.reverse()
        s = values + [d[i]] + h
        print("-".join(s).center(w, "-"))