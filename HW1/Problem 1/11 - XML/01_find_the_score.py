# -*- coding: utf-8 -*-
"""
XML 1 - Find the Score

"""

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    result = 0
    for i in node.iter():
        result += len(i.attrib)

    return result

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))