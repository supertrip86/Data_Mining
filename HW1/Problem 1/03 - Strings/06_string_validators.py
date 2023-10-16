# -*- coding: utf-8 -*-
"""
String Validators

"""

if __name__ == '__main__':
    s = input()

    alnum = any(list(map((lambda x: x.isalnum()), list(s))))
    alpha = any(list(map((lambda x: x.isalpha()), list(s))))
    digit = any(list(map((lambda x: x.isdigit()), list(s))))
    lower = any(list(map((lambda x: x.islower()), list(s))))
    upper = any(list(map((lambda x: x.isupper()), list(s))))

    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)