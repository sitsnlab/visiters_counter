# -*- coding: utf-8 -*-
"""Tkinterを利用したサイズ可変のアプリケーション.

"""


import time
import tkinter as tk
from PIL import Image, ImageTk, ImageOps  # 画像データ用
import cv2


class SimplePicViewer(tk.Frame):
    """app."""

    def __init__(self, frame_reader, master=None, fps: int = 100,
                 startup: bool = False):
        """Init.

        Parameters
        ----------
        frame_reader : 匿名関数
            np.ndarrayを返す画像を読み込む関数.
        master : TYPE, optional
            DESCRIPTION. The default is None.
        fps : int, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
        self.frame_reader = frame_reader
        if master is None:
            master = tk.Tk()
        super().__init__(master)
        self.pack()

        self.master.title("動画表示")       # ウィンドウタイトル
        self.master.geometry("1200x800")     # ウィンドウサイズ(幅x高さ)

        # 上情報
        toparea = tk.Frame(self.master)

        # ボタン系
        self.text = tk.StringVar(value='topmost')
        self.but1 = tk.Button(toparea, command=self.fix_front,
                              textvariable=self.text)

        self.text2 = tk.StringVar(value='Capture')
        self.but2 = tk.Button(toparea, command=self.canvas_click,
                              textvariable=self.text2)

        self.but1.pack(side=tk.LEFT)
        self.but2.pack(side=tk.LEFT)

        # Canvasの作成
        self.canvas = tk.Canvas(self.master)
        # self.canvas.bind('<Button-1>', self.canvas_click)  # マウスイベントの追加

        # fpsの計算
        bottomarea = tk.Frame(self.master)
        label1 = tk.Label(bottomarea, text='FPS : ')
        self.fpsvar = tk.StringVar(value=0.00)
        fpslabel = tk.Label(bottomarea, textvariable=self.fpsvar)
        label1.pack(side=tk.LEFT)
        fpslabel.pack(side=tk.LEFT)

        toparea.pack(side=tk.TOP)
        self.canvas.pack(expand=True, fill=tk.BOTH, side=tk.TOP)
        bottomarea.pack(side=tk.BOTTOM)

        self.disp_id = None
        self.frame_count = 0
        self.fps_interval = 10
        self.oldtime = 0
        self.cycle = int(1000/fps)

        if startup:
            self.canvas_click()

    def fix_front(self):
        """最前面表示."""
        a = not self.master.attributes('-topmost')
        self.master.attributes('-topmost', a)
        self.text.set('topmost' if not a else 'deactivate')

    def canvas_click(self, event=None):
        """Canvasのマウスクリックイベント."""
        if self.disp_id is None:
            # 動画を表示
            self.but2.configure(bg='red')
            self.text2.set('Stop')
            self.input_image()

        else:
            # 動画を停止（予約の停止）
            self.after_cancel(self.disp_id)
            self.disp_id = None
            self.but2.configure(bg='white')
            self.text2.set('Capture')

    def input_image(self):
        """画像をCanvasに表示する."""
        frame = self.frame_reader()

        # BGR→RGB変換 > Pillow.Image
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)

        # キャンバスサイズを取得
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 画像のアスペクト比（縦横比）を崩さずに指定したサイズ（キャンバスのサイズ）全体に画像をリサイズする
        pil_image = ImageOps.pad(pil_image, (canvas_width, canvas_height))

        # PIL.ImageからPhotoImageへ変換する
        self.photo_image = ImageTk.PhotoImage(image=pil_image)

        # 画像の描画
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width/2, canvas_height / 2,
                                 image=self.photo_image)
        self.frame_count += 1

        if self.frame_count == 0:
            self.oldtime = time.time()
        elif self.frame_count > self.fps_interval:
            t = time.time() - self.oldtime
            fps = self.frame_count / t
            self.fpsvar.set(fps)
            self.oldtime, self.frame_count = time.time(), 0

        # input_image()をcycle(msec)後に実行する予約
        self.disp_id = self.after(self.cycle, self.input_image)


if __name__ == "__main__":
    # VideoCapture オブジェクトを取得
    capture = cv2.VideoCapture(0)

    def readcam():
        """カメラデータを取得."""
        _, frame = capture.read()
        return frame

    app = SimplePicViewer(readcam)
    app.mainloop()
