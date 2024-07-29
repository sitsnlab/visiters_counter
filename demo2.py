"""デモ.

"""

# %% 検出
import cv2
from PIL import Image
import time

from processors.mivolo.mivolo_predictor import MiVOLOPredictor
from cv_toolkit.screen_reader import ScreenReader

checkpoint = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
weitght = r".\models\yolov8x_person_face.pt"
read_screen = False
monitor_num = 0
t1 = time.time()
tsum = 0
count = 0

# MiVOLOの初期化
mivolo = MiVOLOPredictor(checkpoint=checkpoint, detector_weights=weitght,
                         disable_faces=False, with_persons=True,
                         draw=True, verbose=False)

# カメラ類の初期化
capture = cv2.VideoCapture(0)
sreader = ScreenReader(monitor_num=monitor_num)

print('Start detection.')
while True:
    """画像検出."""
    if read_screen:
        image = sreader.read_screen()  # スクリーン検出
    else:
        _, image = capture.read()  # カメラ検出

    _, out_im = mivolo.recognize(image)  # 物体検出

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('camera',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('camera', out_im)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    t2 = time.time()
    tsum += t2 - t1
    t1 = t2
    count += 1
    if tsum >= 1:
        fps = count / tsum
        count, tsum = 0, 0
        print(f"fps: {fps}")

capture.release()
cv2.destroyAllWindows()
