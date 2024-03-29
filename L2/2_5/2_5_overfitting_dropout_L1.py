# coding: utf-8
# ## Dropout
# ## Dropout + L1

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common import layers
from data.mnist import load_mnist
import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet
from common import optimizer

(x_train, d_train), (x_test, d_test) = load_mnist (normalize=True)

print ("データ読み込み完了")

# 過学習を再現するために、学習データを削減
x_train = x_train [:300]
d_train = d_train [:300]

# ドロップアウト設定 ======================================
use_dropout = True
dropout_ratio = 0.08
# ====================================================

network = MultiLayerNet (input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                         use_dropout=use_dropout, dropout_ratio=dropout_ratio)

iters_num = 1000
train_size = x_train.shape [0]
batch_size = 100
learning_rate = 0.01

train_loss_list = []
accuracies_train = []
accuracies_test = []
hidden_layer_num = network.hidden_layer_num

plot_interval = 10

# 正則化強度設定 ======================================
weight_decay_lambda = 0.004
# =================================================

for i in range (iters_num):
    batch_mask = np.random.choice (train_size, batch_size)
    x_batch = x_train [batch_mask]
    d_batch = d_train [batch_mask]

    grad = network.gradient (x_batch, d_batch)
    weight_decay = 0

    for idx in range (1, hidden_layer_num + 1):
        grad ['W' + str (idx)] = network.layers ['Affine' + str (idx)].dW + weight_decay_lambda * np.sign (
            network.params ['W' + str (idx)])
        grad ['b' + str (idx)] = network.layers ['Affine' + str (idx)].db
        network.params ['W' + str (idx)] -= learning_rate * grad ['W' + str (idx)]
        network.params ['b' + str (idx)] -= learning_rate * grad ['b' + str (idx)]
        weight_decay += weight_decay_lambda * np.sum (np.abs (network.params ['W' + str (idx)]))

    loss = network.loss (x_batch, d_batch) + weight_decay
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
plt.title ("count - accuracy : Dropout and L1")
plt.xlabel ("count")
plt.ylabel ("accuracy")
plt.ylim (0, 1.0)
# グラフの表示
plt.show ()