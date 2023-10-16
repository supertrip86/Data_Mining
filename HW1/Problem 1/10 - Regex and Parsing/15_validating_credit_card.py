# -*- coding: utf-8 -*-
"""
Validating Credit Card Numbers

"""

import re

a = r'^(4|5|6)([0-9]{3})-?([0-9]{4})-?([0-9]{4})-?([0-9]{4})$'
b = r'([0-9])\1\-?\1\-?\1\-?'

n = int(input())

for i in range(n):
    data = input()
    first = re.match(a, data)
    second = re.findall(b, data)

    if not second and first:
        print("Valid")
    else:
        print("Invalid")