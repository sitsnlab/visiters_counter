# -*- coding: utf-8 -*-
"""各機能を統合したモジュール.

Created on Mon Aug 12 19:40:57 2024.
@author: Yuta Kuronuma
"""
import cv2

from processors.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ImgPadding


if __name__ == '__main__':
    yolo_weight = r".\models\yolov8x_person_face.pt"
    mivolo_weight = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
    reid_weight = r".\models\reid_model_addblock3.pth.tar-22"

    # MiVOLOの初期化
    vc_pred = VCPredictor(yolo_weight, mivolo_weight, reid_weight,
                          draw=True, disable_faces=False, with_persons=True,
                          verbose=False, reid_thrs=30)

    # カメラ類の初期化
    capture = cv2.VideoCapture(0)

    # 画像拡張設定
    ret, frame = capture.read()
    padding = ImgPadding(frame, 0)

    # 画像表示設定（全画面）
    frame_name = 'visitor_counter'
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = capture.read()
        if ret:
            # 位置検出
            results, out_im = vc_pred.recognize(frame, clip_person=False)
            out_im = padding.padding_image(out_im)

            cv2.imshow(frame_name, out_im)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break


    capture.release()
    cv2.destroyAllWindows()
