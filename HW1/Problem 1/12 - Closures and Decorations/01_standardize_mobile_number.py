# -*- coding: utf-8 -*-
"""
Standardize Mobile Number Using Decorators

"""

def wrapper(f):
    def fun(l):
        nums = []
        for i in l:
            nums.append("+91 " + i[-10:][:5] + " " + i[-10:][5:])
        nums.sort()
        for s in nums:
            print(s)

    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 