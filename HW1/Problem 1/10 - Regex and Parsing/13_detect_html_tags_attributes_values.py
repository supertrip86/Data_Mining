# -*- coding: utf-8 -*-
"""
Detect HTML Tags, Attributes and Attribute Values

"""

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if len(attrs):
            for i in attrs:
                print("->", i[0], ">", i[1])
    def handle_startendtag(self, tag, attrs):
        print(tag)
        if len(attrs):
            for i in attrs:
                print("->", i[0], ">", i[1])

parser = MyHTMLParser()

n = int(input())

data = ''

for i in range(n):
    data += input() + "\n"

parser.feed(data)