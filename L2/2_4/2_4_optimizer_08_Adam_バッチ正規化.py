# coding: utf-8
# # optimizer
# ## Adam
# [try] 学習率を変えてみよう
#       learning_rate = 0.01 => 0.03
# [try] 活性化関数と重みの初期化方法を変えてみよう
# 初期状態ではsigmoid - gauss
#   => activationはReLU、weight_init_std:'Xavier'
# [try] バッチ正規化をしてみよう
#   => use_batchnormをTrueにしよう


import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet

# データの読み込み
(x_train, d_train), (x_test, d_test) = load_mnist (normalize=True, one_hot_label=True)

print ("データ読み込み完了")

# batch_normalizationの設定 ================================
# [try] バッチ正規化をしてみよう
use_batchnorm = True
# use_batchnorm = False
# ====================================================
'''  変更箇所
network = MultiLayerNet (input_size=784, hidden_size_list=[40, 20], output_size=10, activation='sigmoid',
                         weight_init_std=0.01,
                         use_batchnorm=use_batchnorm)
'''  # 活性化関数と重みの初期化方法を変えてみよう
network = MultiLayerNet (input_size=784, hidden_size_list=[40, 20], output_size=10, activation='relu',
                         weight_init_std='Xavier',
                         use_batchnorm=use_batchnorm)

iters_num = 1000
train_size = x_train.shape [0]
batch_size = 100
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval = 10

for i in range (iters_num):
    batch_mask = np.random.choice (train_size, batch_size)
    x_batch = x_train [batch_mask]
    d_batch = d_train [batch_mask]

    # 勾配
    grad = network.gradient (x_batch, d_batch)
    if i == 0:
        m = {}
        v = {}
    learning_rate_t = learning_rate * np.sqrt (1.0 - beta2 ** (i + 1)) / (1.0 - beta1 ** (i + 1))
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        if i == 0:
            m [key] = np.zeros_like (network.params [key])
            v [key] = np.zeros_like (network.params [key])

        m [key] += (1 - beta1) * (grad [key] - m [key])
        v [key] += (1 - beta2) * (grad [key] ** 2 - v [key])
        network.params [key] -= learning_rate_t * m [key] / (np.sqrt (v [key]) + 1e-7)

    if (i + 1) % plot_interval == 0:
        accr_test = network.accuracy (x_test, d_test)
        accuracies_test.append (accr_test)
        accr_train = network.accuracy (x_batch, d_batch)
        accuracies_train.append (accr_train)
        loss = network.loss (x_batch, d_batch)
        train_loss_list.append (loss)

        print ('Generation: ' + str (i + 1) + '. 正答率(トレーニング) = ' + str (round (100 * accr_train, 2)) + '%')
        print ('          : ' + str (i + 1) + '. 正答率(テスト) = ' + str (round (100 * accr_test, 2)) + '%')

lists = range (0, iters_num, plot_interval)
plt.plot (lists, accuracies_train, label="training set")
plt.plot (lists, accuracies_test, label="test set")
plt.legend (loc="lower right")
plt.title ("count - accuracy : Adam 3 learning_rate = 0.03 activation=ReLU weight_init_std='Xavier' use_batchnorm = True")
plt.xlabel ("count")
plt.ylabel ("accuracy")
plt.ylim (0, 1.0)
# グラフの表示
plt.show ()
