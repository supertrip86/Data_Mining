# -*- coding: utf-8 -*-
"""
String Formatting

"""

def print_formatted(number):
    w = len(bin(number)[2:]) + 1
    for i in range(1, number + 1):
        a = str(i)
        b = str(oct(i))[2:]
        c = str(hex(i))[2:].upper()
        d = str(bin(i))[2:]

        print(a.rjust(w - 1) + b.rjust(w) + c.rjust(w) + d.rjust(w))