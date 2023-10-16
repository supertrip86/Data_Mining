# -*- coding: utf-8 -*-
"""
Calendar Module

"""

import calendar

inp = list(map(int,input().split()))

week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

weekday = calendar.weekday(inp[2], inp[0], inp[1])

print(week[weekday].upper())