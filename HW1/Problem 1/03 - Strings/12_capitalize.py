# -*- coding: utf-8 -*-
"""
Capitalize!

"""

def solve(s):
    splitted = s.split(' ')
    splits = []
    result = ''
    for el in splitted:
        splits.append(list(el))
    
    for value in splits:
        if len(value) > 0:
            value[0] = value[0].upper()
    
    for value in splits:
        result += ''.join(value) + ' '

    return result