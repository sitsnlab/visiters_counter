# -*- coding: utf-8 -*-
"""検出結果をエクセルに書き出す.

"""

import csv
from pathlib import Path as plib
import pandas as pd
import tqdm
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

HEADER = ['visited_numb', "person_id", "time", "gender", "age"]


class DataWriter:
    """検出結果をCSVに書き出すクラス."""

    def __init__(self, save_path: str = None):

        if save_path is None:
            now = dt.datetime.now()
            save_path = now.strftime('%H_%M_%S_%f') + '.csv'
            self.save_file = plib('data') / save_path

            with open(self.save_file, 'w') as f:
                f.write(','.join(HEADER) + '\n')
        else:
            self.save_file = plib(save_path)

    def weite_file(self, text):
        """データを書き込む."""
        with open(self.save_file, 'a') as f:
            f.write(','.join(text) + '\n')


class DataReader:
    """書き出した結果を読み込むクラス."""

    def __init__(self, file_path: str):

        self.file_path = plib(file_path)
        self.dframe = pd.read_csv(self.file_path, header=0, names=HEADER)
        self.categorize_data()

    def categorize_data(self):
        male = self.dframe[self.dframe['gender'] == 'male']
        self.male_plot = self.time2label(male)

        female = self.dframe[self.dframe['gender'] == 'female']
        self.female_plot = self.time2label(female)

    def time2label(self, gender):
        # '%H:%M:%S.%f'
        # '%Y-%m-%d %H:%M:%S.%f'
        time_rabel = []
        for data in gender['time']:
            h, m, s = list(map(float, data.split(sep=':')))
            # print(h, m, s, h + m/60)
            time_rabel.append(h + m/60)

        return time_rabel


    def show_plot(self):
        # rng = np.random.default_rng()
        # x = rng.standard_normal(1000)
        # y = rng.standard_normal(1000)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist([self.male_plot, self.female_plot], stacked=False)

        plt.show()


if __name__ == '__main__':
    # dw = DataWriter('test.csv')

    # text = ["2", "3", "4", 'Cuda']
    # for i in tqdm.tqdm(range(10)):
    #     dw.weite_file(text)

    # a = 'sasakama'
    # dw.weite_file(a)
    # print(','.join('sasakama'))

    path = r'..\data\output13_58_02_658691.csv'
    dr = DataReader(path)
    dr.categorize_data()
    print(dr.dframe)
    # print(dr.male_plot)
    dr.show_plot()
