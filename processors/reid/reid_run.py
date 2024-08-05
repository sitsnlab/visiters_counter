# -*- coding: utf-8 -*-
"""Re-IDの実行テスト.
Created on Wed Jul 31 19:44:07 2024

@author: ab19109
"""

# %% 検出
import cv2
from PIL import Image
import time

import sys
import os.path as path
from pathlib import Path as plib
# sys.path.append(path.join(path.dirname(__file__), ".."))
# sys.path.append(path.join(path.dirname(__file__), "../.."))

from processors.visitor_predictor import VCPredictor
from cv_toolkit.screen_reader import ScreenReader

from reid_tools import load_model
from myosnet_highres1 import osnet_x1_0 as osnet
from reid_opencampus import ReID

import itertools
import glob

models_path = plib(__file__).parents[2] / 'models'
print(models_path)

# %%
checkpoint = models_path / "model_imdb_cross_person_4.22_99.46.pth.tar"
weitght = models_path / "yolov8x_person_face.pt"
reid_model_path = models_path / 'reid_model_addblock3.pth.tar-22'
image_save_dir = plib(__file__).parents[2] / 'visitor_images'

# checkpoint = r"..\..\models\model_imdb_cross_person_4.22_99.46.pth.tar"
# weitght = r"..\..\models\yolov8x_person_face.pt"
# reid_model_path = r'..\..\models\reid_model_addblock3.pth.tar-22'
# image_save_dir = r'..\..\visitor_images'
read_screen = False
monitor_num = 0
t1 = time.time()
tsum = 0
count = 0

# MiVOLOの初期化
mivolo = VCPredictor(mivolo_weight=checkpoint, yolo_weight=weitght,
                     disable_faces=False, with_persons=True,
                     draw=True, verbose=False)

# カメラ類の初期化
capture = cv2.VideoCapture(0)
sreader = ScreenReader(monitor_num=monitor_num)

#Re-IDクラスのインスタンス
reid = ReID()
reid.image_size = (512, 256)
reid.thrs = 30
reid.save_dir = image_save_dir
#print("save dir > ", path.abspath(r'.\visitor_images'))
#Re-IDモデル
reid_model = osnet(
    num_classes = 1,
    pretrained = False
    )
reid_model = reid_model.cuda()
reid_model.eval()
load_model(reid_model, reid_model_path)
reid.prepare(reid_model)

print('Start detection.')
while True:
    """画像検出."""
    if read_screen:
        image = sreader.read_screen()  # スクリーン検出
    else:
        _, image = capture.read()  # カメラ検出

    detects, out_im = mivolo.recognize(image)  # 物体検出

    #print(detects.md_results[0].person.xyxy)

    '''
    Re-ID実行
    '''
    for i, person in enumerate(detects.md_results):
        detect_result = person.person.xyxy
        people = list(itertools.chain.from_iterable(detect_result.tolist()))
        #print("people > ", people)
        #人物画像作成
        person = image[round(people[1]): round(people[3]), round(people[0]): round(people[2])]


        pid = reid.run_reid(person)
        cv2.putText(person, pid, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
        cv2.imshow(f"{i}_person", person)

    # cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('camera',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
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
print("{} People were detected".format(len(glob.glob(path.join(image_save_dir, '*.jpg')))))
