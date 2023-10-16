# -*- coding: utf-8 -*-
"""
HTML Parser - Part 1

"""

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        if len(attrs):
            for i in attrs:
                print("->", i[0], ">", i[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        if len(attrs):
            for i in attrs:
                print("->", i[0], ">", i[1])

parser = MyHTMLParser()

n = int(input())

data = ''

for i in range(n):
    data += input() + "\n"

parser.feed(data)