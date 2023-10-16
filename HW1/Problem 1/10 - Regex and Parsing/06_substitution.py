# -*- coding: utf-8 -*-
"""
Regex Substitution

"""

import re

n = int(input())

string = ''

for i in range(n):
    string += input() + "\n"

a = re.sub(r'(?<= )&&(?= )', 'and', string)
b = re.sub(r'(?<= )\|\|(?= )', 'or', a)

print(b)