# -*- coding: utf-8 -*-
"""
Re.findall() & Re.finditer()

"""

import re

pattern = r'(?<=[^AEIOUaeiou])[AEIOUaeiou]{2,}(?=[^AEIOUaeiou])'

# (?<=[^AEIOUaeiou]) lookbehind, check if starts with NON vowel
# [AEIOUaeiou]{2,} search for substrings that contain at least two consecutive vowels
# (?=[^AEIOUaeiou]) lookahead, check if end with NON vowel

m = re.findall(pattern, input())

if m:
    for i in m:
        print(i)
else:
    print(-1)