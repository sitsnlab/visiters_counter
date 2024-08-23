# -*- coding: utf-8 -*-
"""YOLOによる物体検出.

https://docs.ultralytics.com/quickstart/#use-ultralytics-with-python

Created on Fri Aug 23 23:09:01 2024.
@author: Yuta Kuronuma
"""

from ultralytics import YOLO
import cv2

from processors.screen_reader import ImgPadding

if __name__ == '__main__':

    # Load a pretrained YOLO model (recommended for training)
    model_name = "yolov5s.pt"
    model_name = "yolov8n.pt"
    model_name = "yolov8x.pt"
    model = YOLO(model_name)

    capture = cv2.VideoCapture(1)

    # 画像拡張, 録画機能の初期化
    ret, frame = capture.read()
    padding = ImgPadding(frame, 0)

    # 全画面表示設定
    frame_name = model_name[:-3]
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = capture.read()
        if ret:
            results = model.predict(source=frame)

            img = results[0].plot()
            img = padding.padding_image(img)

            # 結果表示
            cv2.imshow(frame_name, img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
