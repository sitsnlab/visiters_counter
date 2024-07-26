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


if __name__ == '__main__':
    from simple_picviewer import SimplePicViewer

    # ss = ScreenSelector()
    # print(ss.get_monitor())
    # print(ss.get_bbox(1))
    sreader = ScreenReader(monitor_num=1)

    SPV = SimplePicViewer(sreader.read_screen, startup=True)
    SPV.mainloop()
