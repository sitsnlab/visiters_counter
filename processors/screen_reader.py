# -*- coding: utf-8 -*-
"""画面上の映像を取得する.

"""

import cv2
import numpy as np
from PIL import ImageGrab
import screeninfo
# import pyautogui


class ScreenReader:
    """画面上の映像を取得するクラス."""

    def __init__(self, monitor_num: int = 0):

        screen = ScreenSelector()
        self.bbox = screen.get_bbox(monitor_num)

    def read_screen(self) -> np.ndarray:
        """PC画面を取得し，cv形式で出力する."""
        # スクリーンショット(PIL)
        # left, top, right, bottom
        # img = pyautogui.screenshot()
        img = ImageGrab.grab(bbox=self.bbox, all_screens=True)

        # 形式変換
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img


class ScreenSelector:
    """2枚以上のスクリーンを利用している際の画面指定用クラス."""

    def __init__(self):
        ms = screeninfo.get_monitors()
        self.monitor_list = []

        for i, m in enumerate(ms):
            if m.is_primary:
                self.monitor_list.insert(0, m)
            else:
                self.monitor_list.append(m)

    def get_monitor(self, index: int = 0) -> screeninfo.Monitor:
        """指定した番号のモニターを取得する."""
        if index >= len(self.monitor_list):
            raise IndexError('モニターの数が合いません.')

        return self.monitor_list[index]

    def get_bbox(self, index: int = 0) -> tuple:
        """指定した番号のモニター領域を取得する."""
        m: screeninfo.Monitor = self.get_monitor(index)

        return (m.x, m.y, m.x + m.width, m.y + m.height)


class ImgPadding:
    """スクリーンに合わせて画像を広げるクラス."""

    def __init__(self, img: np.ndarray, screen_num: int = 0):
        selector = ScreenSelector()
        self.monitor = selector.get_monitor(screen_num)

        # 画面比率
        self.screen_ratio = self.monitor.width / self.monitor.height
        self.img_ratio = img.shape[1] / img.shape[0]

        # 画像設定
        shape = img.shape
        pad_size = list(img.shape)
        self.axis = None
        if self.img_ratio <= self.screen_ratio:
            pad_size[0] = shape[0]
            pad_size[1] = int((self.screen_ratio * shape[0] - shape[1]) / 2)
            self.padding_size = [pad_size[0], pad_size[1] * 2 + shape[1]]
            self.axis = 1
        else:
            pad_size[0] = int((shape[1] / self.screen_ratio - shape[0]) / 2)
            pad_size[1] = shape[1]
            self.padding_size = [pad_size[0] * 2 + shape[0], pad_size[1]]
            self.axis = 0

        self.padding = np.zeros(pad_size, dtype='uint8')

    def padding_image(self, img: np.ndarray):
        """画像をスクリーンに合わせてpaddingする."""
        return np.concatenate([self.padding, img, self.padding], self.axis)


if __name__ == '__main__':
    ss = ScreenSelector()
    # sreader = ScreenReader(monitor_num=0)
    # img = sreader.read()

    capture = cv2.VideoCapture(0)
    ret, img = capture.read()

    pad = ImgPadding(img, 0)
    print(pad.screen_ratio)
    print(pad.img_ratio)
    print('img_shape', img.shape)
    print('img_type', img.dtype)
    print('scr_shape', ss.get_monitor(0))

    while True:
        # img = sreader.read()
        ret, img = capture.read()
        img = pad.padding_image(img)
        # cv2.imwrite('padding.jpg', img)

        frame_name = 'screen'
        cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(frame_name, img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
