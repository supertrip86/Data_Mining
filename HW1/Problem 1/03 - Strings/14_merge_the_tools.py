# -*- coding: utf-8 -*-
"""
Merge the Tools!

"""

def merge_the_tools(string, k):
    l = [string[i:i+k] for i in range(0, len(string), k)]

    result = [ ''.join(list(dict.fromkeys(x))) for x in l ]

    for element in result:
        print(element)