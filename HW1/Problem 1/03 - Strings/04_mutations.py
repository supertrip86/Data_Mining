# -*- coding: utf-8 -*-
"""
Mutations

"""

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = ''.join(l)

    return string
