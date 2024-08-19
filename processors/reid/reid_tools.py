# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:18:23 2024

@author: ab19109
オープンキャンパスの人数計測システムでRe-ID関連の細かい処理をまとめるところ
"""
from __future__ import division, print_function, absolute_import
import pickle
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch

import numpy as np
import torchvision.transforms as T
from PIL import Image



from .mymodels.myosnet_highres1 import osnet_x1_0 as osnet_highres1
from .mymodels.myosnet_highres2 import osnet_x1_0 as osnet_highres2
from .mymodels.osnet_base import osnet_x1_0 as osnet
from .mymodels.osnet_part_addblock import osnet_x1_0 as osnet_addblock
from .mymodels.osnet_part_addblock_dellarge import osnet_x1_0 as osnet_addblock_dellarge
from .mymodels.osnet_part_delsmall import osnet_x1_0 as osnet_delsmall




'''
CNNを呼び出す関数
'''
def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    '''
    Parameters
    ----------
    name : str
        CNNの名前
    num_classes : int
        分類クラス数
    loss : str, optional
        損失関数. The default is 'softmax'.
    pretrained : bool, optional
        事前学習済みモデルを使うか. The default is True.
    use_gpu : bool, optional
        GPUを使うか. The default is True.

    Raises
    ------
    KeyError
        DESCRIPTION.

    Returns
    -------
    model : nn.Module
        CNN

    '''

    model_container = {
        'osnet': osnet,
        'osnet_highres1': osnet_highres1,
        'osnet_highres2': osnet_highres2,
        'osnet_addblock': osnet_addblock,
        'osnet_addblock_dellarge': osnet_addblock_dellarge,
        'osnet_delsmall': osnet_delsmall
        }

    model_set = list(model_container.keys())
    if name not in model_set:
        raise KeyError(
            'Unknown model: {}. Model must be one of {}'.format(name, model_set)
            )

    model = model_container[name](
        num_classes = num_classes,
        loss = loss,
        pretained = pretrained,
        use_gpu = use_gpu
        )

    return model


'''
画像から特徴ベクトルを得るクラス
'''
class MyFeatureExtractor(object):
    def __init__(self, model, image_size):

        #パラメータ類
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        device = 'cuda'


        #Transform function
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]

        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        #Class attributes
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.to_pil = to_pil


    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)
            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):

            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():

            features = self.model(images)

        return features



'''
学習済みモデルの読み込みを行う関数
'''
def load_model(model, weight_path):
    '''
    Parameters
    ----------
    model : nn.Module
        CNN
    weight_path : str
        学習済みモデルのパス

    Returns
    -------
    None.

    '''

    #Load checkpoint
    if weight_path is None:
        raise ValueError("File path is None")

    weight_path = osp.abspath(osp.expanduser(weight_path))
    if not osp.exists(weight_path):
        raise FileNotFoundError("File is not found at <{}>".format(weight_path))

    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(weight_path, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = torch.load(weight_path, pickle_module=pickle, map_location=map_location)

    except Exception:
        print("Unable to load checkpoint from <{}>".format(weight_path))
        raise

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers = []
    discarded_layers = []

    for k, v in state_dict.items():
        if k.startswith("module."):
            #discard modules.
            k = k[7:]

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)

        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            "The pretrained weights <{}> cannnot be loaded, "
            "please check the key named manually"
            "(** ignored and continue **".format(weight_path)
        )

    else:
        print("Successfully loaded pretrained weights from <{}>".format(weight_path))
        if len(discarded_layers) > 0:
            print("** The following layers are discrded"
                  "due to unmatched keys or layer size: {}".format(discarded_layers)
              )


'''
特徴ベクトル間のユークリッド距離を計算する関数
'''
def calc_euclidean_dist(input1, input2):
    '''
    Parameters
    ----------
    input1 : Tensor
        画像をCNNに入力して得られる特徴ベクトル
    input2 : Tensor
        画像をCNNに入力して得られる特徴ベクトル

    Returns
    -------
    distmat : Tensor
        特徴ベクトル間のユークリッド距離

    '''

    '''
    計算前の確認
    '''
    #入力がテンソルか確認
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)

    #入力が2次元か確認
    assert input1.dim() == 2, \
        'input1: Expected 2-D tensor, but got {}-D tensor'.format(input1.dim())

    assert input2.dim() == 2, \
        'input2: Expected 2-D tensor, but got {}-D tensor'.format(input2.dim())

    #2つのベクトルのサイズが同じか確認
    assert input1.size(1) == input2.size(1), \
        'Both input must be the same size ({} and {})'.format(input1.size(), input2.size())


    '''
    距離計算
    '''
    m = input1.size(0)
    n = input2.size(0)

    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    distmat = mat1 + mat2

    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)


    return distmat



'''
画像から特徴ベクトルを抽出する関数
'''
def feature_extractor(model, image, image_size):
    '''
    Parameters
    ----------
    model : nn.Module
        CNN
    image : ndarray
        画像
    image_size : tuple
        CNNに入力する画像サイズ(H, W)

    Returns
    -------
    features: Torch.tensor
        特徴ベクトル

    '''


    '''
    パラメータ類
    '''
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    pixel_norm = True
    device = 'cuda'


    #transform関数の作成
    transforms = []
    transforms += [T.Resize(image_size)]
    transforms += [T.ToTensor()]

    if pixel_norm:
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]

    preprocess = T.Compose(transforms)

    to_pil = T.ToPILImage()

    device = torch.device(device)
    model.to(device)

    '''
    前準備
    '''
    if isinstance(image, list):
        images = []

        for element in image:
            if isinstance(element, str):
                image = Image.open(element).convert('RGB')

            elif isinstance(element, np.ndarray):
                image = to_pil(element)

            else:
                raise TypeError(
                    "Type of each element must be belong to [str | numpy.ndarray]"
                )

            image = preprocess(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.to(device)

    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')
        image = preprocess(image)
        images = image.unsqueeze(0).to(device)

    elif isinstance(image, np.ndarray):
        image = to_pil(image)
        image = preprocess(image)
        images = image.unsqueeze(0).to(device)

    elif isinstance(image, torch.Tensor):
        input_image = Image.open(image)
        if input_image.dim() == 3:
            image = image.unsqueeze(0)
        images = image.to(device)

    else:
        raise NotImplementedError

    '''
    特徴抽出
    '''
    with torch.no_grad():
        features = model(images)

    return features



