# coding: utf-8
# ## Dropout
# ## [try] dropout_ratioの値を変更してみよう
#
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from common import optimizer


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand (*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

from common import optimizer

(x_train, d_train), (x_test, d_test) = load_mnist (normalize=True)

print ("データ読み込み完了")

# 過学習を再現するために、学習データを削減
x_train = x_train [:300]
d_train = d_train [:300]

# ドロップアウト設定 ======================================
use_dropout = True
dropout_ratio = 0.005 # 0.15 # [try] dropout_ratioの値を変更してみよう
# ====================================================
# 正則化強度設定 ======================================
weight_decay_lambda = 0.005
# =================================================


network = MultiLayerNet (input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                         weight_decay_lambda=weight_decay_lambda, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
optimizer = optimizer.SGD (learning_rate=0.01)
# optimizer = optimizer.Momentum(learning_rate=0.01, momentum=0.9)
# optimizer = optimizer.AdaGrad(learning_rate=0.01)
# optimizer = optimizer.Adam()

iters_num = 1000
train_size = x_train.shape [0]
batch_size = 100

train_loss_list = []
accuracies_train = []
accuracies_test = []

plot_interval = 10

for i in range (iters_num):
    batch_mask = np.random.choice (train_size, batch_size)
    x_batch = x_train [batch_mask]
    d_batch = d_train [batch_mask]

    grad = network.gradient (x_batch, d_batch)
    optimizer.update (network.params, grad)

    loss = network.loss (x_batch, d_batch)
    train_loss_list.append (loss)

    if (i + 1) % plot_interval == 0:
        accr_train = network.accuracy (x_train, d_train)
        accr_test = network.accuracy (x_test, d_test)
        accuracies_train.append (accr_train)
        accuracies_test.append (accr_test)

        print ('Generation: ' + str (i + 1) + '. 正答率(トレーニング) = ' + str (round (100 * accr_train, 2)) + '%')
        print ('          : ' + str (i + 1) + '. 正答率(テスト) = ' + str (round (100 * accr_test, 2)) + '%')

lists = range (0, iters_num, plot_interval)
plt.plot (lists, accuracies_train, label="training set")
plt.plot (lists, accuracies_test, label="test set")
plt.legend (loc="lower right")
plt.title ("count - accuracy : Dropout dropout_ratio = 0.005")
plt.xlabel ("count")
plt.ylabel ("accuracy")
plt.ylim (0, 1.0)
# グラフの表示
plt.show ()


