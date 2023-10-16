# -*- coding: utf-8 -*-
"""
HTML Parser - Part 2

"""

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):

        if data != "\n" and "\n" in data:
            print(">>> Multi-line Comment")
        elif data != "\n":
            print(">>> Single-line Comment")
        print(data)

    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()