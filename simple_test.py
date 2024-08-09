"""デモ.

"""


import cv2
import time

from processors.visitor_predictor import VCPredictor
from processors.data_IO import DataWriter
from cv_toolkit.screen_reader import ScreenReader

t0 = time.time()
try:
    if __name__ == '__main__':
        yolo_weight = r".\models\yolov8x_person_face.pt"
        mivolo_weight = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
        reid_weight = r".\models\reid_model_addblock3.pth.tar-22"
        save_path = r'.\data\output'
        t1 = time.time()
        count = 0

        # MiVOLOの初期化
        vc_pred = VCPredictor(yolo_weight, mivolo_weight, reid_weight,
                              draw=True, disable_faces=False, with_persons=True,
                              verbose=False)

        # カメラ類の初期化
        capture = cv2.VideoCapture(0)
        # sreader = ScreenReader(monitor_num=1)

        # 書き出し機能の初期化
        dw = DataWriter(save_path)

        while True:
            ret, frame = capture.read()
            # ret, frame = True, sreader.read_screen()  # スクリーン検出
            if ret:
                # 位置検出
                results, out_im = vc_pred.recognize(frame, clip_person=False)

                # 結果記録
                # print(results.md_results[0].dump_data())
                # for md_obj in results.md_results:
                for md_obj in vc_pred.new_visitors:
                    dw.weite_file(md_obj.dump_data().values())
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

        # capture.release()
        # cv2.destroyAllWindows()
except Exception as e:
    print(e)
finally:
    print(time.time() - t0)
    capture.release()
    cv2.destroyAllWindows()
