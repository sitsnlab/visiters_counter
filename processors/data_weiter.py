# -*- coding: utf-8 -*-
"""検出結果をエクセルに書き出す.

"""

import csv

class DataWriter:
    """検出結果をエクセルに書き出すクラス."""

    def __init__(self):
        pass

    @staticmethod
    def weite_file(text):
        with open('input.csv', 'w') as f:
            f.write(','.join(text) + '\n')



if __name__ == '__main__':
    text = ["2","3","4"]

    for i in range(10000):
        DataWriter.weite_file(text)
