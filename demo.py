"""デモ.

"""


import cv2
import argparse

from processors.mivolo.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ScreenReader
from cv_toolkit.simple_picviewer import SimplePicViewer


parser = argparse.ArgumentParser(description='カメラ映像を取り込み，検出した被写体の属性を画面上に描画する．')

parser.add_argument('--checkpoint', help='MiVOLOのチェックポイントパス',
                    default=r".\models\model_imdb_cross_person_4.22_99.46.pth.tar")
parser.add_argument('--detector_weitght', help='YOLOのチェックポイントパス',
                    default=r".\models\yolov8x_person_face.pt")
parser.add_argument('--read_screen', help='pc画面から画像を取り込む.',
                    default=False)
parser.add_argument("--monitor_num", help="読み込むモニター番号",
                    default=0)

args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = args.checkpoint
    detector_weitght = args.detector_weitght
    read_screen = args.read_screen

    # MiVOLOの初期化
    mivolo = VCPredictor(checkpoint=checkpoint, detector_weights=detector_weitght,
                             draw=True, disable_faces=False, with_persons=True, verbose=False)

    # カメラ類の初期化
    capture = cv2.VideoCapture(0)
    sreader = ScreenReader(monitor_num=args.monitor_num)

    def detect():
        """画面の取得，検出."""
        if read_screen:
            image = sreader.read_screen()  # スクリーン検出
        else:
            _, image = capture.read()  # カメラ検出

        _, out_im = mivolo.recognize(image)  # 物体検出

        return out_im

    # アプリケーションの実行
    SPV = SimplePicViewer(detect, startup=True)
    SPV.mainloop()

    capture.release()
