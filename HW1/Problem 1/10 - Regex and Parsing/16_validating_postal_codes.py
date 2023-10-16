# -*- coding: utf-8 -*-
"""
Validating Postal Codes

"""

regex_integer_in_range = r"^[1-9][0-9][0-9][0-9][0-9][0-9]$"
regex_alternating_repetitive_digit_pair = r"([0-9])(?=[0-9]\1)"

# first regex:      from start (^) to end ($) check that the string matches,
#                   in order, a digit from 1 to 9 ([1-9]) and then 5 digits from 0 to 9.
#
# second regex:     The first capturing group ([0-9]) is a digit from 0 to 9, the second is a lookahead that matches a digit 
#                   from 0 to 9 followed by "\1", which is the result of the first capturing group. I used a lookahead to avoid 
#                   skipping characters when the regex scans the string. For instance: in the case of the following RegEx:
#                   ([0-9])([0-9])(\1)
#                   the string "110000" matches the values "11[000]00", but then skips the second result, which would have been 
#                   "110[000]".

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)