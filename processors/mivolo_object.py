# -*- coding: utf-8 -*-
"""モデルの検出結果を保持，表示するためのモジュール.

"""

from copy import deepcopy
import datetime as dt

import cv2
import numpy as np

from ultralytics.engine.results import Results, Boxes
from ultralytics.utils.plotting import Annotator, colors
from mivolo.structures import PersonAndFaceResult


class FrameDetectResult(PersonAndFaceResult):
    """各フレーム毎の検出結果を持つクラス."""

    def __init__(self, results: Results, last_id=0, line_width=None,
                 font_size=None, font="Arial.ttf", pil=False, img=None):
        """イニシャライザ.

        Parameters
        ----------
        results : Results
            YOLOの検出結果.
        last_id : TYPE, optional
            人物IDの初期値. The default is 0.
        line_width : float, optional
            描画するBoundingBoxの線太さ. The default is None.
        font_size : float, optional
            描画するBoundingBoxのフォントサイズ. The default is None.
        font : str, optional
            描画するフォント. The default is "Arial.ttf".
        pil : TYPE, optional
            画像をPillowで描画するか. Falseならばcv(np.ndarray)
            The default is False.
        img : TYPE, optional
            描画先の画像.指定がなければYOLOの検出結果を利用する.
            The default is None.

        Returns
        -------
        None.

        """
        super().__init__(results)

        # アノテータ―の設定(Ultralytics)
        self.img = self.yolo_results.orig_img if img is None else img
        self.names = self.yolo_results.names
        self.annotator = Annotator(deepcopy(self.img), line_width, font_size,
                                   font, pil, example=self.names)

        self.colors_by_ind = {}
        self.md_results: list[MiVOLODetectResult] = []
        self.person_face_id: list[tuple[int, int]] = []
        self.last_id = last_id

    def visitorid_list(self):
        """検出した人物のIDリストを返す."""
        return [person.person_id for person in self.md_results]

    def update_result(self):
        """MiVOLOでの解析後，色と検出した人物の情報を更新する."""
        self.color_bbox()
        self.make_result_list()

    def color_bbox(self):
        """表示する際の色を決定する."""
        # 顔と体の色を合わせて指定する.
        for face_ind, person_ind in self.face_to_person_map.items():
            if person_ind is not None:
                self.colors_by_ind[face_ind] = face_ind + 2
                self.colors_by_ind[person_ind] = face_ind + 2
                self.person_face_id.append((face_ind, person_ind))
            else:
                self.colors_by_ind[face_ind] = 0
                self.person_face_id.append((face_ind, None))

        # 顔のない人の体はピンク
        for person_ind in self.unassigned_persons_inds:
            self.colors_by_ind[person_ind] = 1
            self.person_face_id.append((None, person_ind))

    def make_result_list(self):
        """検出結果からMiVOLODetectResultのリストを作成する.

        顔と体，体のみ⇒検出結果作成
        顔のみ⇒排除
        """
        p_boxes = self.yolo_results.boxes

        for index, (face_ind, person_ind) in enumerate(self.person_face_id):
            if person_ind is None:
                continue
            elif face_ind is None:
                color = self.colors_by_ind[person_ind]
            else:
                color = self.colors_by_ind[face_ind]

            # 性別，年齢の指定
            gender = self.fix_gender(face_ind, person_ind)
            age = self.fix_age(face_ind, person_ind)

            mvdr = MiVOLODetectResult(p_boxes[person_ind], p_boxes[face_ind],
                                      color, gender=gender, age=age)
            self.md_results.append(mvdr)

    def fix_gender(self, face_ind=None, person_ind=None):
        """性別を決定する.

        スコアの高い判定を採用する.

        Parameters
        ----------
        face_ind : TYPE, optional
            顔のインデックス. The default is None.
        person_ind : TYPE, optional
            体のインデックス. The default is None.

        Returns
        -------
        None.

        """
        if (face_ind is not None) and (person_ind is not None):
            if self.gender_scores[face_ind] < self.gender_scores[person_ind]:
                return self.genders[person_ind]
            else:
                return self.genders[face_ind]

        elif face_ind is None:
            return self.genders[person_ind]
        else:
            return self.genders[face_ind]

    def fix_age(self, face_ind=None, person_ind=None):
        """年齢を設定する.

        顔と体の年齢を平均する.

        Parameters
        ----------
        face_ind : TYPE, optional
            DESCRIPTION. The default is None.
        person_ind : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if (face_ind is not None) and (person_ind is not None):
            return (self.ages[face_ind] + self.ages[person_ind]) / 2

        elif face_ind is None:
            return self.ages[person_ind]
        else:
            return self.ages[face_ind]

    def plot_MiVOLODetectResult(self, plot_id=False, plot_info=False):
        """検出した結果をMiVOLODetectResultに基づいて表示する."""
        for mvdr in self.md_results:
            mvdr.plot_box(self.annotator, plot_id, plot_info)

    def plot_textbox(self, xy, text,
                     txt_color=(255, 255, 255), bg_color=(0, 0, 0), space=5):
        """指定した位置にテキストボックスを描画する.

        日本語非対応
        Parameters
        ----------
        xy : TYPE
            描画位置.文字の左上座標
        text : TYPE
            描画メッセージ.
        anchor : TYPE, optional
            描画位置のアンカー. The default is 'top'.
        txt_color : TYPE, optional
            文字色. The default is (255, 255, 255).
        bg_color : TYPE, optional
            背景色. The default is (0, 0, 0).

        Returns
        -------
        None.

        """
        # self.annotator.text(xy, text, anchor=anchor, txt_color=txt_color,
        #                     box_style=True)

        # fontの太さ
        tf = max(self.annotator.lw - 1, 1)  # font thickness

        # 背景の四角形描画
        # テキストの幅，高さ
        w, h = cv2.getTextSize(text, 0, thickness=tf,
                               fontScale=self.annotator.lw / 3)[0]

        p2 = xy[0] + w, xy[1] + h + 2*space

        cv2.rectangle(self.annotator.im, xy, p2, bg_color, -1, cv2.LINE_AA)

        # 文字の描画
        left_bottom = xy[0], xy[1] + h + space
        cv2.putText(self.annotator.im, text, left_bottom, 0,
                    self.annotator.lw / 3, txt_color, thickness=tf,
                    lineType=cv2.LINE_AA)

    def print_info(self):
        """出力情報表示."""
        print('names: \t', self.yolo_results.names)
        print('age: \t', self.ages)
        print('gender: \t', self.genders)
        print('g_score: \t', self.gender_scores)
        print('f-p_map: \t', self.face_to_person_map)
        print('boxes: ', self.yolo_results.boxes.xyxy)
        print('colors: ', self.colors_by_ind)
        print('probs: ', self.yolo_results.probs)
        print('key: ', self.yolo_results.keypoints)
        print()

    def get_img(self) -> np.ndarray:
        """画像を書き出す."""
        return self.annotator.result()


class MiVOLODetectResult():
    """各人物ことのインスタンス."""

    def __init__(self, person: Boxes, face: Boxes, color: int,
                 age: float, gender: str, person_id=None,):
        """イニシャライザ.

        Parameters
        ----------
        person : Boxes
            人物のBoundingBox.
        face : Boxes
            顔のBoundingBox.
        color : int
            描画する際の色.
        age : float
            年齢.
        gender : str
            性別.
        person_id : TYPE, optional
            個人ID. The default is None.

        Returns
        -------
        None.

        """
        self.person: Boxes = self.box_setter(person)
        self.face: Boxes = self.box_setter(face)
        self.color: int = color
        self.age = age
        self.gender = gender
        self.person_id = person_id
        self.visited_numb = 0

        now = dt.datetime.now()
        self.time = now.strftime('%H:%M:%S.%f')
        self.date = now.strftime('%Y/%m/%d')

    def box_setter(self, bbox: Boxes) -> Boxes | None:
        """バウンディングボックスのセッター.

        何らかの形で描画できないボックスが指定された場合はNoneを返す.

        Parameters
        ----------
        bbox : Boxes
            対象のバウンディングボックス.

        Returns
        -------
        None.

        """
        dim = bbox.xyxy.dim()
        return bbox if dim == 2 else None

    def plot_box(self, annotator: Annotator, plot_id=False, plot_info=False,
                 text: str | None = None):
        """指定したアノテータに顔と人をプロットする.

        Parameters
        ----------
        annotator : Annotator
            表示先のアノテータ.
        plot_id : TYPE, optional
            idの表示. The default is False.
        plot_info : TYPE, optional
            性別，年齢を表示するか. The default is False.

        Returns
        -------
        None.

        """
        flabel = ''
        label = f'numb_{self.visited_numb}'
        if plot_id: label += f', {self.person_id}'
        if plot_info: label += f', {self.gender}, {self.age}'
        if text is not None:label += text

        if self.person is not None:
            # print('outperson: ', self.person)
            annotator.box_label(self.person.xyxy.squeeze(), label=label,
                                color=colors(self.color, True))
        else:
            flabel = label

        if self.face is not None:
            annotator.box_label(self.face.xyxy.squeeze(), label=flabel,
                                color=colors(self.color, True))

    def dump_data(self):
        """検出した情報を出力する."""
        data = {}

        data['person_id'] = self.person_id
        data['time'] = self.time
        data['gender'] = self.gender
        data['age'] = self.age
