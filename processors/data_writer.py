# -*- coding: utf-8 -*-
"""検出結果をエクセルに書き出す.

"""

import csv
from pathlib import Path as plib
import pandas as pd
import tqdm
import datetime as dt

class DataWriter:
    """検出結果をCSVに書き出すクラス."""

    def __init__(self, save_path: str):
        now = dt.datetime.now()
        save_path += now.strftime('%H_%M_%S_%f') + '.csv'
        label = ['visited_numb', "person_id", "time", "gender", "age"]

        self.save_file = plib(save_path)
        with open(self.save_file, 'w') as f:
            f.write(','.join(label) + '\n')


    def weite_file(self, text):
        with open(self.save_file, 'a') as f:
            f.write(','.join(text) + '\n')


class DataReader:
    """書き出した結果を読み込むクラス."""



if __name__ == '__main__':
    dw = DataWriter('test.csv')

    text = ["2", "3", "4", 'Cuda']
    for i in tqdm.tqdm(range(10)):
        dw.weite_file(text)

    a = 'sasakama'
    dw.weite_file(a)
    # print(','.join('sasakama'))
