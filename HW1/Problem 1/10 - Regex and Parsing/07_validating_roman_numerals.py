# -*- coding: utf-8 -*-
"""
Validating Roman Numerals

"""

import re

regex_pattern = r'^(M{0,3})(CD|CM|D?C{0,3})(XL|XC|L?X{0,3})(IV|IX|V?I{0,3})$'

# ^(M{0,3})             first group, check if string starts with a substring that matches from 0 to 3 "M"
# (CD|CM|D?C{0,3})      second group, check if characters of second group matches any of the components of column "Hundreds"*
# (XL|XC|L?X{0,3})      third group, check if characters of second group matches any of the components of column "Tens"*
# (IV|IX|V?I{0,3})$     fourth group, check if last group (the one that ends the string) matches any of the components of column "Units"*

# *see table at https://en.wikipedia.org/wiki/Roman_numerals

import re
print(str(bool(re.match(regex_pattern, input()))))