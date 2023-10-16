# -*- coding: utf-8 -*-
"""
Map and Lambda Function

"""

cube = lambda x: x**3

def fibonacci(n):
    result = [0, 1]
    
    if n == 0:
        return []
    elif n == 1:
        return [0]
    else:
        i = n - 2
        while i > 0:
            value = result[-1] + result[-2]
            result.append(value)
            i -= 1
            
        return result

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))