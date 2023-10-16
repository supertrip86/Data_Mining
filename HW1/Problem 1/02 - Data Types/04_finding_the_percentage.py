# -*- coding: utf-8 -*-
"""
Finding the percentage

"""

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    grades = student_marks[query_name]
    average = sum(grades)/len(grades)
    roundedAverage = format(average, '.2f')

    print(roundedAverage)