# -*- coding: utf-8 -*-
"""
Validating UID

"""

n = int(input())

for i in range(n):
    data = input()

    a = len([x for x in data if x.isupper()]) >= 2
    b = len([x for x in data if x.isdigit()]) >= 3
    c = len([x for x in data if x.isalnum()]) == 10
    d = len(set(data)) == len(data)
    e = len(data) == 10
    
    if a and b and c and d and e:
        print("Valid")
    else:
        print("Invalid")