# -*- coding: utf-8 -*-
"""各モデルの検出結果を統合するモジュール.

"""


import time
from typing import Optional, Tuple
import numpy as np
import itertools
import cv2

from ultralytics.engine.results import Results
from mivolo.model.yolo_detector import Detector
from mivolo.model.mi_volo import MiVOLO
from .mivolo_object import FrameDetectResult

from .reid.reid_tools import load_model
from .reid.myosnet_highres1 import osnet_x1_0 as osnet
from .reid.reid_opencampus import ReID

class VCPredictor:
    """人物の検出と同定を行うクラス."""

    def __init__(self, yolo_weight: str, mivolo_weight: str, reid_weight: str,
                 device: str = "cuda:0",
                 with_persons: bool = False, disable_faces: bool = False,
                 draw: bool = False, verbose: bool = False):
        """イニシャライザ.

        Parameters
        ----------
        yolo_weight : str
            YOLOチェックポイントパス.
        mivolo_weight : str
            MiVOLOチェックポイントパス.
        reid_weight : str
            Re-IDチェックポイントパス.
        device : str
            使用デバイス.
        with_persons : bool, optional
            検出に体画像を使用するか. The default is False.
        disable_faces : bool, optional
            顔画像を無視するか. The default is False.
        draw : bool, optional
            検出結果を描画するか. The default is False.
        verbose : bool, optional
            検出情報の詳細を出力するか. The default is False.

        Returns
        -------
        None.

        """
        # モデルの初期化
        self.YOLO_model = Detector(yolo_weight, device, verbose=verbose)
        self.MiVOLO_model = MiVOLO(mivolo_weight, device, verbose=verbose,
                                   half=True, use_persons=with_persons,
                                   disable_faces=disable_faces)

        #Re-IDクラスのインスタンス
        self.reid = ReID()
        self.reid.image_size = (512, 256)
        self.reid.thrs = 10
        self.reid.save_dir = 'visitor_images'
        #Re-IDモデル
        reid_model = osnet(num_classes=1, pretrained=False).cuda()
        reid_model.eval()
        load_model(reid_model, reid_weight)
        self.reid.prepare(reid_model)
        pre_ids = self.reid.dict_gallery_features.keys()

        # visitor計測用
        self.visitor_dict: dict[str:int] = {key: i+1 for i, key in
                                            enumerate(pre_ids)}
        self.visitor_count = len(self.visitor_dict)
        self.new_visitors = []

        # FPS計算用
        self.oldtime = time.time()
        self.count = 0
        self.fps = 0.0

        self.detected_objects: FrameDetectResult = None
        self.old_objects: FrameDetectResult = None

    def recognize(self, image: np.ndarray,
                  plot_id: bool = True, plot_info: bool = True,
                  clip_person: bool = False
                  ) -> Tuple[FrameDetectResult, Optional[np.ndarray]]:
        """1フレームの検出を行う.

        Parameters
        ----------
        image : np.ndarray
            フレーム.
        plot_id : bool, optional
            個人IDを描画するか. The default is True.
        plot_info : bool, optional
            フレーム情報を描画するか. The default is True.

        Returns
        -------
        detected_objects : TYPE
            検出結果クラス.
        out_im : np.ndarray
            検出結果を描画した画像.

        """
        # (YOLO)人物，顔検出
        YOLO_results: Results = self.YOLO_model.yolo.predict(
            image, **self.YOLO_model.detector_kwargs)[0]

        # 検出結果統合クラス
        last_num = len(self.visitor_dict)
        self.detected_objects = FrameDetectResult(YOLO_results, last_num)

        # (MiVOLO)年齢,性別の判定
        self.MiVOLO_model.predict(image, self.detected_objects)
        self.detected_objects.update_result()

        # (ReID)人物同定
        for i, miv_obj in enumerate(self.detected_objects.md_results):
            detect_result = miv_obj.person.xyxy
            people = list(itertools.chain.from_iterable(detect_result.tolist()))
            # print("people > ", people)

            # 人物画像作成
            person = image[round(people[1]): round(people[3]), round(people[0]): round(people[2])]

            # ID付与
            pid = self.reid.run_reid(person)
            miv_obj.person_id = pid

            if clip_person:
                cv2.putText(person, pid, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
                cv2.imshow(f"{i}_person, {pid}", person)

        # 文字情報の更新
        self.update_fps()
        self.update_visitor()
        self.old_objects = self.detected_objects
        # self.detected_objects.print_info()

        # 検出結果の描画
        out_im = self.draw_result(plot_id, plot_info)

        return self.detected_objects, out_im

    def update_fps(self):
        """現在のfpsを計算する."""
        if (times := time.time() - self.oldtime) >= 1:
            fps = self.count / times
            self.fps = round(fps * 100) / 100
            self.oldtime = time.time()
            self.count = 0
        else:
            self.count += 1

    def update_visitor(self):
        """現在の来場者数を更新する."""
        self.new_visitors = []
        for md_obj in self.detected_objects.md_results:
            visitorid = md_obj.person_id
            if self.visitor_dict.get(visitorid) is None:
                self.visitor_count += 1
                self.visitor_dict[visitorid] = self.visitor_count
                self.new_visitors.append(md_obj)
                print('update ! ')

        for miv_obj in self.detected_objects.md_results:
            miv_obj.visited_numb = self.visitor_dict[miv_obj.person_id]

    def draw_result(self, plot_id, plot_info):
        """検出結果の描画."""
        # 人物検出結果を描画
        self.detected_objects.plot_MiVOLODetectResult(plot_id, plot_info)

        # FPS，来場者数を描画
        text = f'fps:{self.fps}, visitor:{self.visitor_count}'
        self.detected_objects.plot_textbox((0, 0), text=text)
        return self.detected_objects.get_img()


def dummy_id(obj):
    """性別と年齢からIDを付与する."""
    num = obj.age // 10 * 10
    did = f"{num}_{obj.gender}"
    return did

