"""デモ.
test
"""

# %% 検出
import cv2
from PIL import Image

from processors.mivolo.mivolo_predictor import MiVOLOPredictor
from cv_toolkit.screen_reader import ScreenReader

checkpoint = r".\models\model_imdb_cross_person_4.22_99.46.pth.tar"
weitght = r".\models\yolov8x_person_face.pt"
read_screen = False
monitor_num = 0

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

    # 結果保存
    # pil_img = Image.fromarray(out_im)
    try:
        cv2.imwrite('temp.jpg', out_im)
        # pil_img.save('temp.jpg')
    except PermissionError:
        continue

capture.release()

# %% 画像読み出し，表示

import cv2
from cv_toolkit.simple_picviewer import SimplePicViewer
import time


# def detect2():
#     """画面の取得，検出."""
#     # 結果読み込み
#     img = cv2.imread('temp.jpg')
#     return img


# # アプリケーションの実行
# SPV = SimplePicViewer(detect2, startup=True)
# SPV.mainloop()

t1 = time.time()
tsum = 0
count = 0

while True:
    t2 = time.time()
    tsum += t2 - t1
    t1 = t2
    count += 1
    if tsum >= 1:
        fps = count / tsum
        count, tsum = 0, 0
        print(f"fps: {fps}")

    try:
        img = cv2.imread('temp.jpg')
        if img is None:
            continue
        # time.sleep(1/20)
    except Exception as e:
        print(e)
        continue
    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('camera',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('camera', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
