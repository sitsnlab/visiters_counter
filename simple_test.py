"""デモ.

"""


import cv2
import time

from processors.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ScreenReader


if __name__ == '__main__':
    checkpoint = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
    detector_weitght = r".\models\yolov8x_person_face.pt"
    t1 = time.time()
    count = 0

    # MiVOLOの初期化
    mivolo = VCPredictor(detector_weitght, checkpoint, draw=True,
                             disable_faces=False, with_persons=True,
                             verbose=False)

    # カメラ類の初期化
    capture = cv2.VideoCapture(0)
    # sreader = ScreenReader(monitor_num=1)

    while True:
        ret, frame = capture.read()
        # ret, frame = True, sreader.read_screen()  # スクリーン検出
        if ret:
            # 位置検出
            results, out_im = mivolo.recognize(frame)

            # results.print()
            cv2.imwrite('temp.jpg', out_im)
            cv2.imshow(__file__, out_im)
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
