# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:06:47 2024
オープンキャンパス用のRe-IDプログラム
@author: ab19109
"""
'''
必要なファイルはreidフォルダにまとめる
テスト用プログラムは別で作る
'''

'''
入力は画像
'''
import torch
# import reid_tools as rt
from .reid_tools import load_model, MyFeatureExtractor, calc_euclidean_dist, feature_extractor, build_model

import os
import os.path as osp

import glob
import tqdm

import numpy as np
import cv2

import datetime

import itertools

try:
    import openpose_processor as opp

except ModuleNotFoundError:
    print("Failed to import openpose")


'''
Re-ID用クラス
'''
class ReID(object):
    def __init__(self, image_size, save_dir, thrs, use_partreid=False, p_thrs=100, pivod_dict=None):
        #入力画像のサイズ
        self.image_size = image_size

        #特徴ベクトルを入れていくリスト
        self.dict_gallery_features = {}
        #人物のIDを入れていくリスト
        self.pid_list = []
        #新たに観測された人物の保存先
        self.save_dir = save_dir
        #全身画像のRe-IDで新たな人物と判断する閾値
        self.thrs = thrs

        #身体部位画像でのRe-IDを実行するか
        self.use_partreid = use_partreid
        #身体部位画像のRe-IDにおける閾値
        self.p_thrs = p_thrs

        #身体部位画像を学習したCNNに関する辞書
        self.pivod_dict = pivod_dict

        #身体部位画像の特徴を入れていく辞書
        self.part_features = {}


    '''
    検索画像の特徴を抽出する関数
    学習済みモデルは事前に読み込んでおく
    '''
    def prepare(self, model):
        '''
        Parameters
        ----------
        model : nn.Module
            CNN

        Returns
        -------
        None.

        '''
        #画像保存先作成
        os.makedirs(self.save_dir, exist_ok=True)

        #特徴抽出器
        self.extractor = MyFeatureExtractor(model, self.image_size)

        '''
        モデル読み込み
        '''
        if self.use_partreid:
            for part in self.pivod_dict.keys():
                model = build_model(self.pivod_dict[part]['model_name'], num_classes=1)
                model = model.cuda()
                model.eval()
                load_model(model, self.pivod_dict[part]['path'])

                self.pivod_dict[part]['model'] = model

        '''
        検索画像取得
        '''
        print("Get gallary images...")
        gimgs = glob.glob(osp.join(self.save_dir, '*.jpg'))

        #検索画像の特徴抽出
        for gimg, i in zip(gimgs, tqdm.tqdm(range(len(gimgs)))):
            #画像のファイル名取得
            gname = osp.basename(osp.splitext(gimg)[0])
            #特徴抽出
            gf = self.extractor(gimg)
            self.dict_gallery_features[gname] = gf

            #身体部位画像の特徴
            if self.use_partreid:
                for part in self.pivod_dict.keys():
                    try:
                        part_img = cv2.imread(osp.join(self.save_dir, 'part', part, '{}_{}.jpg'.format(gname, part)))
                        gpf = feature_extractor(self.pivod_dict[part]['model'], part_img, self.pivod_dict[part]['size'])

                        self.part_features[gname][part] = gpf

                    except FileNotFoundError:
                        pass


        print("Got {} images".format(len(gimgs)))


    '''
    新たに観測された人物を登録する関数
    '''
    def regist_new_person(self, image, part_images = None):
        '''
        Parameters
        ----------
        image : ndarray
            人物画像

        part_images: dict
            身体部位画像

        Returns
        -------
        None.

        '''

        #画像ファイル名を作成(= IDを付与)するために現在時刻を取得
        now_time = datetime.datetime.now()
        #ID付与．観測された時分_秒
        pid = now_time.strftime("%H%M_%S")

        #既にIDが存在していた場合(同時に複数人が観測されたとき)
        if pid in self.pid_list:
            #末尾に番号を割り当て，新規IDとなるまで数値を増やす．
            i = 1
            while True:
                new_pid = pid + '_{}'.format(i)
                if new_pid not in self.pid_list:
                    pid = new_pid

                    break

                else:
                    i += 1

        self.pid_list.append(pid)

        #画像の特徴抽出，特徴ベクトル保存
        gf = self.extractor(image)
        self.dict_gallery_features[pid] = gf

        #身体部位画像の保存
        if self.use_partreid:
            for part in self.pivod_dict.keys():
                try:
                    cv2.imwrite(osp.join(self.save_dir, 'part', part, '{}_{}.jpg'.format(pid, part)), part_images[part])

                except:
                    pass

        #画像保存
        cv2.imwrite(osp.join(self.save_dir, pid + '.jpg'), image)
        #print("savede images at {}".format(osp.join(self.save_dir, pid + '.jpg')))

        return pid


    '''
    身体部位画像を使ったRe-ID
    '''
    def part_reid(self, image, gid):
        '''
        Parameters
        ----------
        image : ndarray
            カメラ画像

        gid: str
            全身画像のRe-IDで同一人物と判断された人物のID

        Returns
        -------
        pid: str
            人物ID

        '''

        #身体部位ごとの画像作成
        part_images = opp.crop_by_part(image)

        #部位名をキー，特徴ベクトル間の距離を値とする辞書
        pd_dict = {}

        #身体部位ごとのループ
        for part in self.pivod_dict.keys():
            pd_dict[part] = None

            #部位画像の特徴抽出
            if part_images[part] != None and self.part_features[gid][part] != None:
                cv2.imshow(part, part_images[part])
                cv2.waitKey(1)
                qpf = feature_extractor(self.pivod_dict[part]['model'], part_images[part], self.pivod_dict[part]['size'])
                #検索画像の特徴ベクトルとの距離計算
                dist = calc_euclidean_dist(qpf, self.part_features[gid][part])

                #辞書の値を更新
                pd_dict[part] = dist.item() * self.pivod_dict[part]['weight']

        #辞書の値からNoneを除去
        values = [v for v in list(pd_dict.values) if v != None]

        #加重平均
        factor = np.mean(values)

        #平均値が閾値を下回った場合は両者を同一人物とする
        if factor < self.p_thrs:
            pid = gid

        #閾値を上回った場合は異なる人物とする
        else:
            pid = 'another'


        return pid






    '''
    Re-IDを実行する関数
    '''
    def run_reid(self, qimg, frame):
        '''
        Parameters
        ----------
        qimg : ndarray
            人物画像(1人分)

        frame : ndarray
            カメラ画像

        Returns
        -------
        qid: str
            人物のID
        '''

        #検索画像のIDと特徴ベクトルのリスト
        gf_list = list(self.dict_gallery_features.values())
        gid_list = list(self.dict_gallery_features.keys())

        if len(gf_list) == 0:
            pid = self.regist_new_person(qimg)

            return pid

        '''
        全身画像でのRe-ID
        '''
        #入力画像の特徴抽出
        qf = self.extractor(qimg)
        gfs = [gf.clone().detach() for gf in gf_list]
        gfs = torch.cat(gfs, dim=0)

        #検索画像の特徴ベクトルとの距離計算
        distmat = calc_euclidean_dist(qf, gfs)
        distmat = distmat.cpu().numpy()
        #余分な[]を外す
        dist_iter = list(itertools.chain.from_iterable(distmat))
        print("distmat > , ", dist_iter)
        #距離が短い順にしたときに各要素が元々どの位置にいたかを表すリスト

        #indices = np.argsort(dist_iter, axis=1)
        indice = dist_iter.index(min(dist_iter))
        print("indice > ", indice)

        #距離の最小値が閾値以上の場合は新規人物として登録
        print("min distance > ", dist_iter[indice])
        if dist_iter[indice] > self.p_thrs:
            pid = self.regist_new_person(qimg)

        else:
            pid = gid_list[indice]


        '''
        身体部位画像でのRe-ID
        '''
        if self.use_partreid:
            print("Run part Re-ID")
            for _ in range(len(self.pid_list)):
                pid = self.part_reid(frame, pid)

                if pid != 'another':
                    break

        return pid








