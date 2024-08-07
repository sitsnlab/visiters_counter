# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:25:21 2024

@author: ab19109

カメラ映像を記録する
"""
import cv2
import datetime
import os
import os.path as osp



class Recorder:

    def __init__(self, width: int, height: int, fps: int):
        '''
        Parameters
        ----------
        width : int
            フレームの幅
        height : int
            フレームの高さ
        fps : int
            動画のフレームレート

        Returns
        -------
        None.

        '''
        self.width = width
        self.height = height
        self.fps = fps

    
    def prepare(self, save_dir):
        '''
        Parameters
        ----------
        save_dir : str
            動画保存先

        Returns
        -------
        None.

        '''
        
        date = datetime.datetime.now().strftime('%m%d_%H%M')
        os.makedirs(save_dir, exist_ok=True)
        
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(osp.join(save_dir, date + '.mp4'), fmt, self.fps, (self.width, self.height))
        
    
    def write(self, frame):
        '''
        Parameters
        ----------
        frame : ndarray
            フレーム

        Returns
        -------
        None.

        '''
        
        self.writer.write(frame)
        
        
    def release(self):
        self.writer.release()
        
    



