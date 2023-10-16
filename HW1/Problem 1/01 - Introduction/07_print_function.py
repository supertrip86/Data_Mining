# -*- coding: utf-8 -*-
"""
Print Function

"""

if __name__ == '__main__':
    n = int(input())

    result = ""

    for i in range(n):
        value = str(i + 1)
        result += value

    print(result)