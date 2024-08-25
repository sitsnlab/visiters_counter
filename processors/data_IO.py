# -*- coding: utf-8 -*-
"""検出結果のCSV書き出し，読み込み.

"""

import csv
from pathlib import Path as plib
import pandas as pd
import tqdm
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HEADER = ['visited_numb', "person_id", "time", "gender", "age"]


class DataWriter:
    """検出結果をCSVに書き出すクラス."""

    def __init__(self, save_path: str = None):
        """イニシャライザ.


        Parameters
        ----------
        save_path : str, optional
            保存先のファイル. The default is None.

        Returns
        -------
        None.

        """

        if save_path is None:
            now = dt.datetime.now()
            save_path = now.strftime('%Y_%m%d_%H%M') + '.csv'
            self.save_file = plib('data') / save_path

            with open(self.save_file, 'w') as f:
                f.write(','.join(HEADER) + '\n')
        else:
            self.save_file = plib(save_path)

        self.write_time()

    def write_file(self, text):
        """データを書き込む."""
        with open(self.save_file, 'a') as f:
            f.write(','.join(text) + '\n')

    def write_time(self):
        """データの時間をつける."""
        data = {}

        data['visited_numb'] = '-1'
        data['person_id'] = "None"
        data['time'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        data['gender'] = 'None'
        data['age'] = '-1'

        self.write_file(data.values())


class DataReader:
    """書き出した結果を読み込むクラス."""

    def __init__(self, file_path: str):

        self.file_path = plib(file_path)
        self.dframe = pd.read_csv(self.file_path, header=0, names=HEADER)
        self.categorize_data()
        self.make_title()

    def categorize_data(self):
        male = self.dframe[self.dframe['gender'] == 'male']
        self.male_plot = self.time2label(male)

        female = self.dframe[self.dframe['gender'] == 'female']
        self.female_plot = self.time2label(female)

    def time2label(self, gender):
        # '%Y-%m-%d %H:%M:%S.%f'
        time_rabel = []
        for data in gender['time']:
            data = data.split(sep=' ')[1]
            h, m, s = list(map(float, data.split(sep=':')))
            # time_rabel.append(h + m/60)
            time_rabel.append(h)

        return time_rabel

    def make_title(self):
        date, time = self.dframe['time'][0].split(sep=' ')
        male = len(self.dframe[self.dframe['gender'] == 'male'])
        female = len(self.dframe[self.dframe['gender'] == 'female'])

        times = self.dframe['time'].to_list()
        self.time_s = int(times[0].split(sep=' ')[1].split(sep=':')[0])
        self.time_e = int(times[-1].split(sep=' ')[1].split(sep=':')[0])

        form = 'Visitor of {}, time:{}~{}\n'.format(date, self.time_s, self.time_e)
        form += 'Male:{}, Female:{}'.format(male, female)
        self.title = form.format(date, male, female)

    def show_plot(self):
        bins = self.time_e - self.time_s + 1

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist([self.male_plot, self.female_plot],
                stacked=False, range=(self.time_s, self.time_e), bins=bins)

        fig.suptitle(self.title)
        ax.legend(["Male", "Female"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

        plt.show()


if __name__ == '__main__':
    # dw = DataWriter('test.csv')

    # text = ["2", "3", "4", 'Cuda']
    # for i in tqdm.tqdm(range(10)):
    #     dw.weite_file(text)

    # a = 'sasakama'
    # dw.weite_file(a)
    # print(','.join('sasakama'))

    path = r'..\data\2024_0823_0349.csv'
    path = r'..\data\2024_0823_0349.csv'

    path = r'..\data\2024_0823_0349.csv'
    path = r'..\data\2024_0825_pm.csv'

    dr = DataReader(path)
    dr.categorize_data()
    dr.show_plot()
