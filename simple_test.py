"""各ブランチの追加機能をテストするモジュール."""


import cv2
import time

from processors.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ScreenReader
from cv_toolkit.screen_reader import ImgPadding


if __name__ == '__main__':
    yolo_weight = r".\models\yolov8x_person_face.pt"
    mivolo_weight = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
    reid_weight = r".\models\reid_model_addblock3.pth.tar-22"
    t1 = time.time()
    count = 0

    # MiVOLOの初期化
    vc_pred = VCPredictor(yolo_weight, mivolo_weight, reid_weight,
                          draw=True, disable_faces=False, with_persons=True,
                          verbose=False)
    # vc_pred = None

    # カメラ類の初期化
    capture = cv2.VideoCapture(0)
    # sreader = ScreenReader(monitor_num=1)
    ret, frame = capture.read()

    padding = ImgPadding(frame, 0)

    while True:
        ret, frame = capture.read()
        # ret, frame = True, sreader.read_screen()  # スクリーン検出
        if ret:
            # 位置検出
            results, out_im = vc_pred.recognize(frame, clip_person=False)
            out_im = padding.padding_image(out_im)

            # results.print()
            # cv2.imwrite('temp.jpg', out_im)

            frame_name = 'simple_test'
            cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(frame_name, out_im)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

            count += 1

        if (times := time.time() - t1) >= 1:
            print(count / times)
            t1 = time.time()
            count = 0
        # break

    capture.release()
    cv2.destroyAllWindows()
