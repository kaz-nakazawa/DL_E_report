# coding: utf-8
# # simple convolution network
# ## image to column
 ## [try] im2colの処理を確認しよう
# ・関数内でtransposeの処理をしている行をコメントアウトして下のコードを実行してみよう<br>

import sys, os
sys.path.append (os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common import layers
from common import optimizer
from data.mnist import load_mnist
import matplotlib.pyplot as plt

# 画像データを２次元配列に変換
'''
input_data: 入力値
filter_h: フィルターの高さ
filter_w: フィルターの横幅
stride: ストライド
pad: パディング
'''

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # N: number, C: channel, H: height, W: width
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad (input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros ((N, C, filter_h, filter_w, out_h, out_w))

    for y in range (filter_h):
        y_max = y + stride * out_h
        for x in range (filter_w):
            x_max = x + stride * out_w
            col [:, :, y, x, :, :] = img [:, :, y:y_max:stride, x:x_max:stride]


#   col = col.transpose (0, 4, 5, 1, 2,3) # (N, C, filter_h, filter_w, out_h, out_w) -> (N, filter_w, out_h, out_w, C, filter_h)
    col = col.reshape (N * out_h * out_w, -1)
    return col

# im2colの処理確認
input_data = np.random.rand (2, 1, 4, 4) * 100 // 1  # number, channel, height, widthを表す
print ('========== input_data ===========\n', input_data)
print ('==============================')
filter_h = 3
filter_w = 3
stride = 1
pad = 0
col = im2col (input_data, filter_h=filter_h, filter_w=filter_w, stride=stride, pad=pad)
print ('============= col ==============\n', col)
print ('==============================')
