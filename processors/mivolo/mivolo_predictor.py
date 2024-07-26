# -*- coding: utf-8 -*-
"""MiVOLOを利用した検出を行うクラス.

"""


from typing import Optional, Tuple
import numpy as np
import time

from ultralytics.yolo.engine.results import Results
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector

from .mivolo_object import FrameDetectResult


class MiVOLOPredictor:
    """MiVOLOを使用するためのクラス."""

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
        self.detector = Detector(detector_weights, device, verbose=verbose)
        self.age_gender_model = MiVOLO(checkpoint, device, verbose=verbose,
                                       half=True, use_persons=with_persons,
                                       disable_faces=disable_faces)
        self.draw = draw

        self.count = 0
        self.oldtime = time.time()
        self.fps = 0.0
        self.visitor_count = 0
        self.visitor_list = []

        self.detected_objects: FrameDetectResult = None
        self.old_objects: FrameDetectResult = None

    def recognize(self, image: np.ndarray) -> Tuple[FrameDetectResult,
                                                    Optional[np.ndarray]]:
        """フレーム毎の検出.

        Parameters
        ----------
        image : np.ndarray
            フレーム.

        Returns
        -------
        detected_objects : TYPE
            検出結果クラス.
        out_im : TYPE
            検出結果を表記した画像.

        """
        # 人物，顔検出
        results: Results = self.detector.yolo.predict(
            image, **self.detector.detector_kwargs)[0]
        self.detected_objects = FrameDetectResult(results,
                                                  last_id=len(self.visitor_list))

        # 年齢，性別の判定，結果インスタンスの更新
        self.age_gender_model.predict(image, self.detected_objects)
        self.detected_objects.update_result()

        # 文字情報の更新
        self.update_fps()
        self.update_visitor()
        self.old_objects = self.detected_objects

        # 結果の描画
        out_im = self.draw_result(self.detected_objects) if self.draw else None
        # self.detected_objects.print_info()

        return self.detected_objects, out_im

    def draw_result(self, detected_objects: FrameDetectResult):
        """検出結果の描画."""
        # detected_objects.plot(conf=False, labels=True, show_boxes=True)
        detected_objects.plot_MiVOLODetectResult(plot_id=True, plot_info=True)

        text = f'fps:{self.fps}, visitor:{self.visitor_count}'
        detected_objects.plot_textbox((0, 0), text=text,
                                      txt_color=(255, 255, 255),
                                      bg_color=(0, 0, 0))
        return detected_objects.get_img()

    def update_fps(self):
        """現在のfpsを計算する."""
        interval = 10

        if self.count >= interval:
            fps = self.count / (time.time() - self.oldtime)
            self.fps = round(fps * 100) / 100
            self.count = 0
            self.oldtime = time.time()
        else:
            self.count += 1

    def update_visitor(self):
        """現在の来場者数を更新する."""
        self.visitor_list += self.detected_objects.visitors
        self.visitor_count = len(self.visitor_list)
