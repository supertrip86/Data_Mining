# -*- coding: utf-8 -*-
"""
Group(), Groups() & Groupdict()

"""

import re

pattern = r'([0-9a-zA-Z])\1+' # Matches any single letter or digit, \1+ selects first group with at least one repetions of the given character

m = re.search(pattern, input())

if m:
    print(m.group(0)[0])
else:
    print(-1)