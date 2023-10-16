# -*- coding: utf-8 -*-
"""
Validating phone numbers

"""

import re

pattern = r'^(7|8|9)(\d{9})$'

# ^(7|8|9) the string has to start with either 7, 8 or 9
# (\d{9}) the remaining part of the string has to be a digit (\d) of exactly 9 characters ({9})

n = int(input())

for i in range(n):
    m = re.match(pattern, input())
    
    if m:
        print('YES')
    else:
        print('NO') 