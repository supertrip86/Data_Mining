# -*- coding: utf-8 -*-
"""
The Minion Game

"""

def minion_game(string):
    a = 0
    b = 0

    for i in range(len(string)):
        if string[i] in 'AEIOU':
            a += len(string) - i
        else:
            b += len(string) - i


    if a > b:
        print("Kevin " + str(a))
    elif a < b:
        print("Stuart " + str(b))
    else:
        print("Draw")
