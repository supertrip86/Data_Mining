# -*- coding: utf-8 -*-
"""
Nested Lists

"""

if __name__ == '__main__':

    students = []
    grades = []
    result = []

    for _ in range(int(input())):
        name = input()
        score = float(input())

        students.append([name, score])
        grades.append(score)

    grades.sort()
    grades = list(dict.fromkeys(grades))
    secondLowestGrade = grades[1]

    for sublist in students:
        if sublist[1] == secondLowestGrade:
            result.append(sublist)
            
    result.sort(key = lambda x:x[0])
            
    for value in result:
        print(value[0],sep='\n')