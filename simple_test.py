"""デモ.

"""


import cv2
import time

from processors.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ScreenReader

from processors import recorder


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

    # カメラ類の初期化
    capture = cv2.VideoCapture(0)
    # sreader = ScreenReader(monitor_num=1)

    # 録画準備
    # 動画保存先
    video_dir = r'.\video'
    # フレームの幅．高さ，動画FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = capture.get(cv2.CAP_PROP_FPS)
    fps = 10
    recorder = recorder.Recorder(width, height, fps)
    recorder.prepare(video_dir)

    while True:
        ret, frame = capture.read()
        # ret, frame = True, sreader.read_screen()  # スクリーン検出
        if ret:
            # 位置検出
            results, out_im = vc_pred.recognize(frame, clip_person=True)

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
        
    
        #　動画書込み
        recorder.write(out_im)

    capture.release()
    cv2.destroyAllWindows()
    recorder.release()
