# -*- coding: utf-8 -*-
"""
Hex Color Code

"""

import re

pattern = r'#[0-9ABCDEFabcdef]{6}|#[0-9ABCDEFabcdef]{3}(?=;)'

n = int(input())

data = ''

for i in range(n):
    data += input() + "\n"

m = re.findall(pattern, data)

for i in m:
    print(i)