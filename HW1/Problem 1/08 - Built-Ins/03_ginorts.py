# -*- coding: utf-8 -*-
"""
ginortS

"""

s = input()

ucase = []
lcase = []
even = []
odd = []

for c in s:
    if c.islower():
        lcase.append(c)
    elif c.isupper():
        ucase.append(c)
    elif c.isdigit() and (int(c) % 2 == 0):
        even.append(c)
    elif c.isdigit() and (int(c) % 2 != 0):
        odd.append(c)
        
a = "".join(sorted(lcase))
b = "".join(sorted(ucase))
c = "".join(sorted(odd))
d = "".join(sorted(even))

print(a + b + c + d)