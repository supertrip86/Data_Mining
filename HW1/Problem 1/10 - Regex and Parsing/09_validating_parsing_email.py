# -*- coding: utf-8 -*-
"""
Validating and Parsing Email Addresses

"""

import re

pattern = r'(?<=<)([a-z])([\w\.-]+)@([a-z]+)\.([a-z]{1,3})(?=>)'

# (?<=<)        at the beginning, has to match (but do not include) "<" character
# ([a-z])       first group: username must start with one alphabetic character
# ([\w\.-]+)    second group: the rest of the username can contain alphanumeric character
# @             "@" comes after username
# ([a-z]+)      third group: domain. Only english alphabetical characters ("+" checks that preceding character appears one or more times)
# \.            a dot (.) comes after the domain
# ([a-z]{1,3})  fourth group: the email extension. Only alphabetic characters, minimum one, max three ( "{1, 3}" ")
# (?=>)         at the end, has to match (but do not include) ">" character

n = int(input())

for i in range(n):
    a = input()
    m = re.search(pattern, a)

    if m:
        print(a)