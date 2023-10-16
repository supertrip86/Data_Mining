# -*- coding: utf-8 -*-
"""
Exceptions

"""

T = int(input())

for i in range(T):
    values = input().split()

    try:
        print(int(values[0]) // int(values[1]))
    except ZeroDivisionError as e:
        print("Error Code:",e)
    except ValueError as e:
        print("Error Code:",e)