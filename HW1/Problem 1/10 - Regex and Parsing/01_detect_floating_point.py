# -*- coding: utf-8 -*-
"""
Detect Floating Point Number

"""

import re

pattern = r'^[+-]?[0-9]*\.[0-9]+$'

x = int(input())

for i in range(x):
    if re.match(pattern, input()):
        print(True)
    else:
        print(False)