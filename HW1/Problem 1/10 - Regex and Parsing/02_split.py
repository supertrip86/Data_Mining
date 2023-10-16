# -*- coding: utf-8 -*-
"""
Re.split()

"""

# \D <---> Uppercase D. Matches any character that is not a decimal digit.

regex_pattern = r"\D"

import re
print("\n".join(re.split(regex_pattern, input())))