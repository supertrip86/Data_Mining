# -*- coding: utf-8 -*-
"""
Lists

"""

if __name__ == '__main__':
    N = int(input())

    result = []

    for d in range(N - 1):
        elements = input().split()

        if len(elements) == 3:
            command = "result." + elements[0] + "(" + elements[1] + "," + elements[2] + ")"

        elif len(elements) == 2:
            command = "result." + elements[0] + "(" + elements[1] + ")"

        elif len(elements) == 1:
            if elements[0] == 'print':
                command = "print(result)"
            else:
                command = "result." + elements[0] + "()"

        eval(command)

    print(result)
