# -*- coding: utf-8 -*-
"""各モデルの検出結果を統合するモジュール.

"""


import time
from typing import Optional, Tuple
import numpy as np

from ultralytics.engine.results import Results
from mivolo.model.yolo_detector import Detector

from mivolo.model.mi_volo import MiVOLO

from .mivolo_object import FrameDetectResult


class VCPredictor:
    """人物の検出と同定を行うクラス."""

    def __init__(self, detector_weights: str, checkpoint: str, device: str = "cuda:0",
                 with_persons: bool = False, disable_faces: bool = False,
                 draw: bool = False, verbose: bool = False):
        """イニシャライザ.

        Parameters
        ----------
        detector_weights : str
            YOLOチェックポイントパス.
        checkpoint : str
            MiVOLOチェックポイントパス.
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
        self.YOLO_model = Detector(detector_weights, device, verbose=verbose)
        self.MiVOLO_model = MiVOLO(checkpoint, device, verbose=verbose,
                                   half=True, use_persons=with_persons,
                                   disable_faces=disable_faces)

        # FPS計算用
        self.oldtime = time.time()
        self.count = 0
        self.fps = 0.0

        # visitor計測用
        self.visitor_count = 0
        self.visitor_list: list[str] = []  # 来場者の個人IDリスト

        self.detected_objects: FrameDetectResult = None
        self.old_objects: FrameDetectResult = None

    def recognize(self, image: np.ndarray,
                  plot_id: bool = True, plot_info: bool = True
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
        last_num = len(self.visitor_list)
        self.detected_objects = FrameDetectResult(YOLO_results, last_num)

        # (MiVOLO)年齢,性別の判定
        self.MiVOLO_model.predict(image, self.detected_objects)
        self.detected_objects.update_result()

        # (ReID)人物同定

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
        self.visitor_list += self.detected_objects.visitors
        self.visitor_count = len(self.visitor_list)

    def draw_result(self, plot_id, plot_info):
        """検出結果の描画."""
        # 人物検出結果を描画
        self.detected_objects.plot_MiVOLODetectResult(plot_id, plot_info)

        # FPS，来場者数を描画
        text = f'fps:{self.fps}, visitor:{self.visitor_count}'
        self.detected_objects.plot_textbox((0, 0), text=text)
        return self.detected_objects.get_img()
