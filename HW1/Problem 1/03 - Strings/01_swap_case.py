# -*- coding: utf-8 -*-
"""
sWAP cASE

"""

def swap_case(s):
    newString = ''

    for i in range(len(s)):
        if s[i].isupper():
            newString += s[i].lower()
        else:
            newString += s[i].upper()

    return newString