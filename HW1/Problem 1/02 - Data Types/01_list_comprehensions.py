# -*- coding: utf-8 -*-
"""
List Comprehensions

"""

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    values = [
        [a for a in range(x + 1)],
        [b for b in range(y + 1)],
        [c for c in range(z + 1)]
    ]

    permutations = [[a,b,c] for a in values[0] for b in values[1] for c in values[2] ]

    result = []
            
    for list in permutations:
        if sum(list) != n:
            result.append(list)
            
    print(result)